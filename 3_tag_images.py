# -*- coding: utf-8 -*-
import os
import argparse
import sys
from collections import Counter
# Импорт общих утилит
try:
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr); sys.exit(1)

# --- Функция тегирования ---
def tag_images(images_folder, kohya_dir, venv_dir, method="wd14", batch_size=8, threshold=0.35,
               caption_ext=".txt", blacklist_tags="", min_len=10, max_len=75, overwrite=False):
    """Тегирует изображения с помощью WD14 Tagger или BLIP."""
    print("\n--- Image Tagging ---")
    # Получаем путь к python из venv через утилиту
    python_executable = common_utils.get_venv_python(os.path.dirname(venv_dir), os.path.basename(venv_dir)) # Передаем base_dir и venv_name
    if not python_executable: print("[!] Cannot run tagging scripts.", file=sys.stderr); return

    if not os.path.isdir(images_folder): print(f"[!] Image directory not found: {images_folder}. Skipping.", file=sys.stderr); return

    # Проверка существующих тегов
    try:
        existing_tags = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
        if existing_tags and not overwrite: print(f"[*] Found {len(existing_tags)} existing tag files ('{caption_ext}'). Skipping."); print("    Use --overwrite-tags to re-tag."); return
        if overwrite and existing_tags:
             print(f"[*] --overwrite-tags enabled. Removing {len(existing_tags)} existing tag files...")
             removed_count = 0
             for tag_file in existing_tags:
                  try: os.remove(os.path.join(images_folder, tag_file)); removed_count += 1;
                  except OSError as e: print(f"  [!] Error removing {tag_file}: {e}", file=sys.stderr);
             print(f"[*] Removed {removed_count} files.")
    except OSError as e: print(f"[!] Error accessing image directory {images_folder}: {e}. Skipping.", file=sys.stderr); return

    # Выполнение теггера
    if method == "wd14":
        script_path = os.path.join(kohya_dir, "finetune", "tag_images_by_wd14_tagger.py")
        model_repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v3"
        if not os.path.exists(script_path): print(f"[!] WD14 Tagger script not found: {script_path}. Skipping.", file=sys.stderr); return
        print(f"[*] Running WD14 Tagger (model: {model_repo}, threshold: {threshold})...")
        model_cache_dir = os.path.join(kohya_dir, "wd14_models_cache"); os.makedirs(model_cache_dir, exist_ok=True)
        cmd = [ python_executable, script_path, images_folder, "--repo_id", model_repo, "--model_dir", model_cache_dir, "--thresh", str(threshold), "--batch_size", str(batch_size), "--caption_extension", caption_ext, "--remove_underscore", "--onnx", "--recursive" ]
        common_utils.run_cmd(cmd, check=True) # Используем общую утилиту
        # Пост-обработка: блеклист
        print("[*] Post-processing WD14 tags (applying blacklist)...")
        blacklisted_tags = set(common_utils.split_tags(blacklist_tags)) # Используем общую утилиту
        if not blacklisted_tags: print("[*] No blacklist tags specified."); return
        top_tags = Counter(); processed_files = 0
        try:
            tag_files = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
            for txt_file in tag_files:
                filepath = os.path.join(images_folder, txt_file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: current_tags = common_utils.split_tags(f.read())
                    processed_tags = [tag for tag in current_tags if tag not in blacklisted_tags]
                    top_tags.update(processed_tags)
                    if len(processed_tags) != len(current_tags):
                        with open(filepath, 'w', encoding='utf-8') as f: f.write(", ".join(processed_tags));
                        processed_files += 1
                except Exception as e: print(f"  [!] Error post-processing {txt_file}: {e}", file=sys.stderr)
            print(f"[+] WD14 Tagging complete. Applied blacklist to {processed_files} files.")
            if top_tags: print("  Top 50 tags (after blacklist):"); [print(f"    {tag} ({count})") for tag, count in top_tags.most_common(50)];
        except OSError as e: print(f"[!] Error accessing dir for post-processing: {e}", file=sys.stderr)

    elif method == "blip":
        script_path = os.path.join(kohya_dir, "finetune", "make_captions.py")
        if not os.path.exists(script_path): print(f"[!] BLIP captioning script not found: {script_path}. Skipping.", file=sys.stderr); return
        print(f"[*] Running BLIP Captioning (min: {min_len}, max: {max_len})...")
        cmd = [ python_executable, script_path, images_folder, "--batch_size", "1", "--min_length", str(min_len), "--max_length", str(max_len), "--caption_extension", caption_ext, "--max_data_loader_n_workers", "2" ]
        common_utils.run_cmd(cmd, check=True)
        print("[+] BLIP Captioning complete.")
    else:
        print(f"[!] Unknown tagging method: {method}. Skipping.")

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 3: Automatically tag images.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory.")
    parser.add_argument("--kohya-dir-name", type=str, default="kohya_ss", help="Kohya scripts directory name.")
    parser.add_argument("--venv-name", type=str, default="lora_env", help="Venv directory name.")
    parser.add_argument("--tagging-method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    parser.add_argument("--tagger-threshold", type=float, default=0.35, help="Confidence threshold for WD14.")
    parser.add_argument("--tagger-batch-size", type=int, default=8, help="Batch size for WD14.")
    parser.add_argument("--blip-min-length", type=int, default=10, help="Min caption length for BLIP.")
    parser.add_argument("--blip-max-length", type=int, default=75, help="Max caption length for BLIP.")
    parser.add_argument("--caption-extension", type=str, default=".txt", help="Extension for tag/caption files.")
    parser.add_argument("--tagger-blacklist", type=str, default="", help="Comma-separated tags to blacklist.")
    parser.add_argument("--overwrite-tags", action='store_true', help="Overwrite existing tag files.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")
    kohya_dir = os.path.join(base_dir, args.kohya_dir_name)
    venv_dir = os.path.join(base_dir, args.venv_name) # Путь к папке venv

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