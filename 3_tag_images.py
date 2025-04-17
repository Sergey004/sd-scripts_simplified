# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import sys
import re # Нужен для split_tags
from collections import Counter

# --- Константы ---
SUPPORTED_TAG_TYPES = (".txt", ".caption") # Используется для проверки

# --- Утилиты ---
def run_cmd(command, check=True, shell=False, capture_output=False, text=False, cwd=None, env=None):
    """Утилита для запуска команд оболочки."""
    # ... (полный код run_cmd как в setup_environment.py) ...
    command_str = ' '.join(command) if isinstance(command, list) else command
    print(f"[*] Running: {command_str}" + (f" in {cwd}" if cwd else ""))
    try:
        process = subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text, cwd=cwd, env=env, encoding='utf-8', errors='ignore')
        if capture_output:
             stdout = process.stdout.strip() if process.stdout else ""; stderr = process.stderr.strip() if process.stderr else ""
             if stdout: print(f"  [stdout]:\n{stdout}");
             if stderr: print(f"  [stderr]:\n{stderr}", file=sys.stderr);
        return process
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running command: {command_str}", file=sys.stderr); print(f"[!] Exit code: {e.returncode}", file=sys.stderr);
        if capture_output:
            stdout = e.stdout.decode(errors='ignore').strip() if isinstance(e.stdout, bytes) else (e.stdout.strip() if e.stdout else ""); stderr = e.stderr.decode(errors='ignore').strip() if isinstance(e.stderr, bytes) else (e.stderr.strip() if e.stderr else "")
            if stdout: print(f"[!] Stdout:\n{stdout}", file=sys.stderr);
            if stderr: print(f"[!] Stderr:\n{stderr}", file=sys.stderr);
        if check: print(f"[X] Critical command failed. Exiting."); sys.exit(1);
        return None
    except FileNotFoundError as e:
        print(f"[!] Error: Command not found - {command[0]}. Is it installed and in PATH?", file=sys.stderr); print(f"[!] Details: {e}", file=sys.stderr);
        if check: print(f"[X] Critical command not found. Exiting."); sys.exit(1);
        return None
    except Exception as e:
        print(f"[!] An unexpected error occurred running command '{command_str}': {e}", file=sys.stderr);
        if check: print(f"[X] Unexpected error during critical command. Exiting."); sys.exit(1);
        return None

def split_tags(tagstr):
    """Разделяет строку тегов по запятой."""
    if not isinstance(tagstr, str): return []
    return [s.strip() for s in tagstr.split(",") if s.strip()]

# --- Функция тегирования ---
def tag_images(images_folder, kohya_dir, venv_dir, method="wd14", batch_size=8, threshold=0.35,
               caption_ext=".txt", blacklist_tags="", min_len=10, max_len=75, overwrite=False):
    """Тегирует изображения с помощью WD14 Tagger или BLIP."""
    print("\n--- Image Tagging ---")
    python_executable = os.path.join(venv_dir, 'bin', 'python') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'python.exe')
    if not os.path.exists(python_executable):
        print(f"[!] Python executable not found in venv: {python_executable}. Cannot run tagging.", file=sys.stderr)
        return

    if not os.path.isdir(images_folder):
        print(f"[!] Image directory not found: {images_folder}. Skipping tagging.", file=sys.stderr)
        return

    # Проверка существующих тегов
    try:
        existing_tags = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
        if existing_tags and not overwrite:
            print(f"[*] Found {len(existing_tags)} existing tag files ('{caption_ext}'). Skipping tagging.")
            print("    Use --overwrite_tags to re-tag.")
            return
        if overwrite and existing_tags:
             print(f"[*] --overwrite_tags enabled. Removing {len(existing_tags)} existing tag files...")
             # ... (код удаления файлов) ...
             removed_count = 0
             for tag_file in existing_tags:
                  try: os.remove(os.path.join(images_folder, tag_file)); removed_count += 1;
                  except OSError as e: print(f"  [!] Error removing {tag_file}: {e}", file=sys.stderr);
             print(f"[*] Removed {removed_count} files.")
    except OSError as e:
        print(f"[!] Error accessing image directory {images_folder}: {e}. Skipping tagging.", file=sys.stderr)
        return

    # Выполнение теггера
    if method == "wd14":
        script_path = os.path.join(kohya_dir, "finetune", "tag_images_by_wd14_tagger.py")
        model_repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
        if not os.path.exists(script_path): print(f"[!] WD14 Tagger script not found: {script_path}. Skipping.", file=sys.stderr); return
        print(f"[*] Running WD14 Tagger (model: {model_repo}, threshold: {threshold})...")
        model_cache_dir = os.path.join(kohya_dir, "wd14_models_cache"); os.makedirs(model_cache_dir, exist_ok=True)
        cmd = [ python_executable, script_path, images_folder, "--repo_id", model_repo, "--model_dir", model_cache_dir, "--thresh", str(threshold), "--batch_size", str(batch_size), "--caption_extension", caption_ext, "--force_download", "--remove_underscore" ]
        run_cmd(cmd, check=True)
        # Пост-обработка: блеклист
        print("[*] Post-processing WD14 tags (applying blacklist)...")
        blacklisted_tags = set(split_tags(blacklist_tags))
        if not blacklisted_tags: print("[*] No blacklist tags specified."); return
        top_tags = Counter(); processed_files = 0
        try:
            tag_files = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
            for txt_file in tag_files:
                filepath = os.path.join(images_folder, txt_file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: current_tags = split_tags(f.read())
                    processed_tags = [tag for tag in current_tags if tag not in blacklisted_tags]
                    top_tags.update(processed_tags)
                    if len(processed_tags) != len(current_tags):
                        with open(filepath, 'w', encoding='utf-8') as f: f.write(", ".join(processed_tags));
                        processed_files += 1
                except Exception as e: print(f"  [!] Error post-processing {txt_file}: {e}", file=sys.stderr)
            print(f"[+] WD14 Tagging complete. Applied blacklist to {processed_files} files.")
            if top_tags:
                print("  Top 50 tags (after blacklist):");
                for tag, count in top_tags.most_common(50): print(f"    {tag} ({count})");
        except OSError as e: print(f"[!] Error accessing dir for post-processing: {e}", file=sys.stderr)

    elif method == "blip":
        script_path = os.path.join(kohya_dir, "finetune", "make_captions.py")
        if not os.path.exists(script_path): print(f"[!] BLIP captioning script not found: {script_path}. Skipping.", file=sys.stderr); return
        print(f"[*] Running BLIP Captioning (min: {min_len}, max: {max_len})...")
        cmd = [ python_executable, script_path, images_folder, "--batch_size", "1", "--min_length", str(min_len), "--max_length", str(max_len), "--caption_extension", caption_ext, "--max_data_loader_n_workers", "2" ]
        run_cmd(cmd, check=True)
        print("[+] BLIP Captioning complete.")
    else:
        print(f"[!] Unknown tagging method: {method}. Skipping.")

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 3: Automatically tag images using WD14 or BLIP.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing project, kohya_ss, venv.")
    parser.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name of the kohya_ss directory.")
    parser.add_argument("--venv_name", type=str, default="lora_env", help="Name of the virtual environment directory.")
    parser.add_argument("--tagging_method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    parser.add_argument("--tagger_threshold", type=float, default=0.35, help="Confidence threshold for WD14.")
    parser.add_argument("--tagger_batch_size", type=int, default=8, help="Batch size for WD14.")
    parser.add_argument("--blip_min_length", type=int, default=10, help="Min caption length for BLIP.")
    parser.add_argument("--blip_max_length", type=int, default=75, help="Max caption length for BLIP.")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Extension for tag/caption files.")
    parser.add_argument("--tagger_blacklist", type=str, default="", help="Comma-separated tags to remove after WD14 tagging.")
    parser.add_argument("--overwrite_tags", action='store_true', help="Overwrite existing tag/caption files.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")
    kohya_dir = os.path.join(base_dir, args.kohya_dir_name)
    venv_dir = os.path.join(base_dir, args.venv_name)

    print("--- Step 3: Tag Images ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Image Folder: {images_folder}")
    print(f"[*] Method: {args.tagging_method}")

    tag_images(
        images_folder, kohya_dir, venv_dir, args.tagging_method, args.tagger_batch_size,
        args.tagger_threshold, args.caption_extension, args.tagger_blacklist,
        args.blip_min_length, args.blip_max_length, args.overwrite_tags
    )
    print("\n--- Step 3 Finished ---")