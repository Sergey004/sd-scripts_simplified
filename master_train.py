# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import sys
import platform
import time # Добавим для возможной задержки, если нужно

# Импортируем общие утилиты
try:
    # Ищем common_utils.py в той же директории, что и master_train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir) # Добавляем директорию скрипта в путь поиска
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr)
    print(f"[-] Please ensure common_utils.py is in the same directory as master_train.py ({script_dir}).", file=sys.stderr)
    sys.exit(1)
finally:
    # Убираем добавленный путь, чтобы не мешать другим импортам
    if script_dir in sys.path:
        sys.path.remove(script_dir)


# --- Функция для запуска дочернего скрипта с паузой ---
def run_stage_script(script_name, python_executable, args_dict, check=True, wait_for_enter=True):
    """Запускает указанный скрипт, передавая аргументы из словаря."""
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"[!] Error: Script '{script_name}' not found at {script_path}", file=sys.stderr)
        if check: sys.exit(1)
        return False

    # Формируем список аргументов из словаря
    args_list = []
    for key, value in args_dict.items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value: args_list.append(arg_name)
        elif value is not None:
            args_list.extend([arg_name, str(value)])

    command = [python_executable, script_path] + args_list
    print(f"\n--- Running Stage: {script_name} ---")
    result = common_utils.run_cmd(command, check=check) # Используем run_cmd из утилит

    success = result is not None and result.returncode == 0

    if success:
        print(f"--- Stage '{script_name}' Finished Successfully ---")
        # Пауза только если шаг успешен И паузы включены И это не последний шаг (логика последнего шага в main)
        if wait_for_enter:
             try: input("<?> Press Enter to continue, or Ctrl+C to abort...")
             except KeyboardInterrupt: print("\n[!] Aborted by user."); sys.exit(0)
    else:
        print(f"[!] Stage '{script_name}' failed or finished with errors.", file=sys.stderr)
        # Если check=False, даем шанс продолжить
        if not check:
             try: input("<?> Stage failed (non-critical). Press Enter to continue anyway, or Ctrl+C to abort...")
             except KeyboardInterrupt: print("\n[!] Aborted by user after non-critical failure."); sys.exit(0)
        # Если check=True, run_cmd уже вызвал sys.exit(1)

    return success


# --- Парсер аргументов для мастер-скрипта ---
def parse_master_arguments():
    parser = argparse.ArgumentParser(description="Master script to run LoRA training pipeline stages.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Основные Настройки и Пути ---
    g_main = parser.add_argument_group('Main Settings')
    g_main.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    g_main.add_argument("--base_dir", type=str, default=".", help="Base directory for project, kohya_ss, venv.")
    g_main.add_argument("--venv_name", type=str, default="lora_env", help="Name of the venv directory.")
    g_main.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name of the kohya_ss directory.")

    # --- Управление Этапами ---
    g_steps = parser.add_argument_group('Pipeline Steps Control')
    g_steps.add_argument("--run_steps", type=str, default="setup,scrape,dedupe,tag,curate,config,train", help="Comma-separated list of steps to run (e.g., 'tag,curate,config,train'). Steps: setup, scrape, dedupe, tag, curate, config, train.")
    g_steps.add_argument("--skip_steps", type=str, default="", help="Comma-separated list of steps to skip (e.g., 'scrape,dedupe'). Overrides --run_steps.")
    g_steps.add_argument("--no_wait", action='store_true', help="Do not wait for Enter key between steps.")
    g_steps.add_argument("--skip_initial_pause", action='store_true', help="Skip the initial pause for dataset preparation.") # Новый флаг

    # --- Собираем ВСЕ аргументы для ВСЕХ этапов ---
    # ... (все аргументы из g_scrape, g_dedupe, g_tag, g_curate, g_config, g_run_train как в предыдущем ответе) ...
    # Аргументы scrape (1_scrape_images.py)
    g_scrape = parser.add_argument_group('Step 1: Scrape Images Args')
    g_scrape.add_argument("--scrape_tags", type=str, default="", help="Tags for Gelbooru scraping.")
    g_scrape.add_argument("--scrape_limit", type=int, default=1000, help="Max images to fetch.")
    g_scrape.add_argument("--scrape_max_res", type=int, default=3072, help="Max image resolution for scraping.")
    g_scrape.add_argument("--scrape_include_parents", action=argparse.BooleanOptionalAction, default=True, help="Include Gelbooru posts with parents.")
    # Аргументы dedupe (2_detect_duplicates.py)
    g_dedupe = parser.add_argument_group('Step 2: Detect Duplicates Args')
    g_dedupe.add_argument("--dedup_threshold", type=float, default=0.985, help="Similarity threshold for duplicates.")
    # Аргументы tag (3_tag_images.py)
    g_tag = parser.add_argument_group('Step 3: Tag Images Args')
    g_tag.add_argument("--tagging_method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    g_tag.add_argument("--tagger_threshold", type=float, default=0.35, help="Confidence threshold for WD14.")
    g_tag.add_argument("--tagger_batch_size", type=int, default=8, help="Batch size for WD14.")
    g_tag.add_argument("--blip_min_length", type=int, default=10, help="Min caption length for BLIP.")
    g_tag.add_argument("--blip_max_length", type=int, default=75, help="Max caption length for BLIP.")
    g_tag.add_argument("--tagger_blacklist", type=str, default="bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber", help="Tags to blacklist after WD14 tagging.")
    g_tag.add_argument("--overwrite_tags", action='store_true', help="Overwrite existing tag files during tagging.")
    # Аргументы curate (4_curate_tags.py)
    g_curate = parser.add_argument_group('Step 4: Curate Tags Args')
    g_curate.add_argument("--caption_extension", type=str, default=".txt", help="Extension for tag/caption files.")
    g_curate.add_argument("--activation_tag", type=str, default="", help="Activation tag(s) to prepend.")
    g_curate.add_argument("--remove_tags", type=str, default="lowres, bad anatomy, worst quality, low quality", help="Tags to remove during curation.")
    g_curate.add_argument("--search_tags", type=str, default="", help="Tags to search for replacement.")
    g_curate.add_argument("--replace_tags", type=str, default="", help="Tags to replace with.")
    g_curate.add_argument("--sort_tags_alpha", action='store_true', help="Sort tags alphabetically.")
    g_curate.add_argument("--remove_duplicate_tags", action='store_true', help="Remove duplicate tags.")
    # Аргументы config & train (5_generate_configs.py, 6_run_training.py)
    g_config = parser.add_argument_group('Step 5 & 6: Training Configuration Args')
    g_config.add_argument("--base_model", type=str, required=True, help="REQUIRED: URL or local path to the base model.")
    g_config.add_argument("--custom_model", type=str, default=None, help="Override base_model (alternative).")
    g_config.add_argument("--base_vae", type=str, default=None, help="URL or local path for an external VAE (optional).")
    g_config.add_argument("--custom_vae", type=str, default=None, help="Override base_vae (alternative).")
    g_config.add_argument("--v_pred", action='store_true', help="Set if the base model uses v-prediction.")
    g_config.add_argument("--force_download_model", action='store_true', help="Force download model (in generate_configs step).")
    g_config.add_argument("--force_download_vae", action='store_true', help="Force download VAE (in generate_configs step).")
    g_config.add_argument("--resolution", type=int, default=1024, help="Training resolution.")
    g_config.add_argument("--shuffle_tags", action=argparse.BooleanOptionalAction, default=True, help="Shuffle caption tags.")
    g_config.add_argument("--keep_tokens", type=int, default=1, help="Tokens to keep at caption start.")
    g_config.add_argument("--flip_aug", action='store_true', help="Enable flip augmentation.")
    g_config.add_argument("--num_repeats", type=int, default=None, metavar='N', help="Repeats per image (overrides --auto_repeats).")
    g_config.add_argument("--auto_repeats", action='store_true', help="Auto-determine repeats.")
    g_config.add_argument("--preferred_unit", type=str, choices=["Epochs", "Steps"], default="Epochs", help="Unit for training duration.")
    g_config.add_argument("--how_many", type=int, default=10, help="Number of epochs or steps.")
    g_config.add_argument("--save_every_n_epochs", type=int, default=1, metavar='N', help="Save checkpoint every N epochs.")
    g_config.add_argument("--keep_only_last_n_epochs", type=int, default=10, metavar='N', help="Keep only the last N saved epochs.")
    g_config.add_argument("--caption_dropout", type=float, default=0.0, metavar='RATE', help="Caption dropout rate.")
    g_config.add_argument("--tag_dropout", type=float, default=0.0, metavar='RATE', help="Tag dropout rate.")
    g_config.add_argument("--unet_lr", type=float, default=3e-4, help="U-Net learning rate.")
    g_config.add_argument("--text_encoder_lr", type=float, default=6e-5, help="Text Encoder learning rate.")
    g_config.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", choices=["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"], help="Learning rate scheduler.")
    g_config.add_argument("--lr_scheduler_num_cycles", type=int, default=3, help="Cycles for cosine_with_restarts.")
    g_config.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power for polynomial scheduler.")
    g_config.add_argument("--lr_warmup_ratio", type=float, default=0.05, help="LR warmup ratio.")
    g_config.add_argument("--min_snr_gamma", type=float, default=8.0, help="Min SNR gamma.")
    g_config.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset.")
    g_config.add_argument("--ip_noise_gamma", type=float, default=0.0, help="Instance Prompt Noise Gamma.")
    g_config.add_argument("--multinoise", action='store_true', help="Enable multiresolution noise.")
    g_config.add_argument("--zero_terminal_snr", action='store_true', help="Enable zero terminal SNR.")
    g_config.add_argument("--lora_type", type=str, choices=["LoRA", "LoCon"], default="LoRA", help="Type of LoRA network.")
    g_config.add_argument("--network_dim", type=int, default=32, help="Network dimension (rank).")
    g_config.add_argument("--network_alpha", type=int, default=16, help="Network alpha.")
    g_config.add_argument("--conv_dim", type=int, default=16, help="Conv dimension for LoCon.")
    g_config.add_argument("--conv_alpha", type=int, default=8, help="Conv alpha for LoCon.")
    g_config.add_argument("--continue_from_lora", type=str, default=None, metavar='PATH', help="Path to existing LoRA to continue.")
    g_config.add_argument("--auto_vram_params", action='store_true', help="Auto-set batch_size, optimizer, precision.")
    g_config.add_argument("--train_batch_size", type=int, default=None, metavar='N', help="Batch size (overrides auto).")
    g_config.add_argument("--cross_attention", type=str, choices=["sdpa", "xformers"], default="sdpa", help="Cross attention implementation.")
    g_config.add_argument("--precision", type=str, choices=["float", "fp16", "bf16", "full_fp16", "full_bf16"], default=None, metavar='TYPE', help="Training precision (overrides auto).")
    g_config.add_argument("--cache_latents", action=argparse.BooleanOptionalAction, default=True, help="Cache latents.")
    g_config.add_argument("--cache_latents_to_disk", action=argparse.BooleanOptionalAction, default=True, help="Cache latents to disk.")
    g_config.add_argument("--cache_text_encoder_outputs", action='store_true', help="Cache text encoder outputs.")
    g_config.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing.")
    g_config.add_argument("--optimizer", type=str, default=None, choices=["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"], metavar='OPT', help="Optimizer (overrides auto).")
    g_config.add_argument("--use_recommended_optimizer_args", action='store_true', help="Use recommended args for optimizer.")
    g_config.add_argument("--optimizer_args", type=str, default="", help="Additional optimizer arguments.")
    g_config.add_argument("--max_data_loader_n_workers", type=int, default=2, help="Workers for data loading.")
    g_config.add_argument("--seed", type=int, default=42, help="Random seed.")
    g_config.add_argument("--lowram", action='store_true', help="Enable Kohya low RAM optimizations.")
    g_config.add_argument("--bucket_reso_steps", type=int, default=64, help="Steps for bucket resolution.")
    g_config.add_argument("--min_bucket_reso", type=int, default=256, help="Minimum bucket resolution.")
    g_config.add_argument("--max_bucket_reso", type=int, default=4096, help="Maximum bucket resolution.")
    g_config.add_argument("--bucket_no_upscale", action='store_true', help="Disable upscaling aspect ratio buckets.")
    # Аргументы из 6_run_training.py
    g_run_train = parser.add_argument_group('Step 6: Run Training Args')
    g_run_train.add_argument("--num_cpu_threads", type=int, default=2, help="CPU threads per process for Accelerate.")

    return parser.parse_args()

# --- Основная Логика ---
def main():
    master_args = parse_master_arguments()
    master_args_dict = vars(master_args).copy()

    # Определяем пути
    base_dir = os.path.abspath(master_args.base_dir)
    venv_dir = os.path.join(base_dir, master_args.venv_name)
    kohya_dir = os.path.join(base_dir, master_args.kohya_dir_name)
    project_dir = os.path.join(base_dir, master_args.project_name)
    paths = { "project": project_dir, "images": os.path.join(project_dir, "dataset"), "output": os.path.join(project_dir, "output"), "logs": os.path.join(project_dir, "logs"), "config": os.path.join(project_dir, "config"), "kohya": kohya_dir, "venv": venv_dir }

    # Определяем путь к Python внутри venv
    venv_python = common_utils.get_venv_python(base_dir, master_args.venv_name)

    # Определяем, какие шаги выполнять
    all_steps = ["setup", "scrape", "dedupe", "tag", "curate", "config", "train"]
    steps_to_run = set(s.strip() for s in master_args.run_steps.lower().split(',') if s.strip())
    steps_to_skip = set(s.strip() for s in master_args.skip_steps.lower().split(',') if s.strip())
    steps_to_run = steps_to_run - steps_to_skip

    wait_between_steps = not master_args.no_wait
    skip_initial_pause = master_args.skip_initial_pause # Используем новый флаг

    print("--- Master Script Start ---")
    print(f"Project: {master_args.project_name}"); print(f"Base Dir: {base_dir}"); print(f"Wait between steps: {wait_between_steps}"); print(f"Skip initial pause: {skip_initial_pause}"); print("-" * 20)
    print(f"Steps to run: {', '.join(sorted(list(steps_to_run)))}"); print("-" * 20)

    # --- Создание папок проекта ---
    print("[*] Ensuring project directories exist...")
    for key in ["project", "images", "output", "logs", "config"]:
        try: os.makedirs(paths[key], exist_ok=True)
        except OSError as e: print(f"[X] CRITICAL ERROR: Could not create directory {paths[key]}: {e}", file=sys.stderr); sys.exit(1)

    # ===> НАЧАЛЬНАЯ ПАУЗА <===
    if not skip_initial_pause:
        print("-" * 20)
        print(f"<?> Please prepare your dataset now.")
        print(f"    Ensure your images are inside: {paths['images']}")
        if "tag" not in steps_to_run and "curate" not in steps_to_run:
             print(f"    Ensure your tag files ({master_args.caption_extension}) are also in that folder and correctly named.")
        try: input("<?> Press Enter when ready to proceed, or Ctrl+C to abort...")
        except KeyboardInterrupt: print("\n[!] Aborted by user during initial pause."); sys.exit(0)
        print("-" * 20)
    else:
        print("[*] Skipping initial dataset preparation pause.")

    # Добавим проверку на пустой датасет, если планируются шаги обработки/тренировки
    steps_requiring_images = {"tag", "curate", "config", "train"}
    if not steps_to_run.isdisjoint(steps_requiring_images): # Если есть пересечение
        image_count = common_utils.get_image_count(paths['images'])
        if image_count == 0:
            print(f"[X] CRITICAL ERROR: Dataset folder '{paths['images']}' is empty, but subsequent steps require images.", file=sys.stderr)
            print("[-] Please add images to the dataset folder or adjust the steps to run.")
            sys.exit(1)

    # --- Определяем аргументы для каждого скрипта ---
    script_args_map = {
        "0_setup_environment.py": ['base_dir', 'venv_name', 'kohya_dir_name'],
        "1_scrape_images.py": ['project_name', 'base_dir', 'scrape_tags', 'scrape_limit', 'scrape_max_res', 'scrape_include_parents'],
        "2_detect_duplicates.py": ['project_name', 'base_dir', 'dedup_threshold'],
        "3_tag_images.py": ['project_name', 'base_dir', 'kohya_dir_name', 'venv_name', 'tagging_method', 'tagger_threshold', 'tagger_batch_size', 'blip_min_length', 'blip_max_length', 'caption_extension', 'tagger_blacklist', 'overwrite_tags'],
        "4_curate_tags.py": ['project_name', 'base_dir', 'caption_extension', 'activation_tag', 'remove_tags', 'search_tags', 'replace_tags', 'sort_tags_alpha', 'remove_duplicate_tags'],
        "5_generate_configs.py": ['project_name', 'base_dir', 'base_model', 'custom_model', 'base_vae', 'custom_vae', 'v_pred', 'resolution', 'shuffle_tags', 'keep_tokens', 'flip_aug', 'num_repeats', 'auto_repeats', 'preferred_unit', 'how_many', 'save_every_n_epochs', 'keep_only_last_n_epochs', 'caption_dropout', 'tag_dropout', 'unet_lr', 'text_encoder_lr', 'lr_scheduler', 'lr_scheduler_num_cycles', 'lr_scheduler_power', 'lr_warmup_ratio', 'min_snr_gamma', 'noise_offset', 'ip_noise_gamma', 'multinoise', 'zero_terminal_snr', 'lora_type', 'network_dim', 'network_alpha', 'conv_dim', 'conv_alpha', 'continue_from_lora', 'auto_vram_params', 'train_batch_size', 'cross_attention', 'precision', 'cache_latents', 'cache_latents_to_disk', 'cache_text_encoder_outputs', 'gradient_checkpointing', 'optimizer', 'use_recommended_optimizer_args', 'optimizer_args', 'max_data_loader_n_workers', 'seed', 'lowram', 'bucket_reso_steps', 'min_bucket_reso', 'max_bucket_reso', 'bucket_no_upscale', 'caption_extension'],
        "6_run_training.py": ['project_name', 'base_dir', 'kohya_dir_name', 'venv_name', 'num_cpu_threads']
    }

    # --- Запуск этапов ---
    current_step_num = 0
    steps_to_run_ordered = [s for s in all_steps if s in steps_to_run] # Сохраняем порядок

    for step_name in steps_to_run_ordered:
        script_file = ""; stage_args_dict = {}; python_to_use = venv_python; is_critical = True; is_last_step = (step_name == steps_to_run_ordered[-1])

        if step_name == "setup": script_file = "0_setup_environment.py"; python_to_use = sys.executable; is_critical = True
        elif step_name == "scrape": script_file = "1_scrape_images.py"; is_critical = False
        elif step_name == "dedupe": script_file = "2_detect_duplicates.py"; is_critical = False
        elif step_name == "tag": script_file = "3_tag_images.py"; is_critical = True
        elif step_name == "curate": script_file = "4_curate_tags.py"; is_critical = True
        elif step_name == "config": script_file = "5_generate_configs.py"; is_critical = True
        elif step_name == "train": script_file = "6_run_training.py"; is_critical = True

        current_step_num += 1
        print(f"\n>>> === Starting Pipeline Step {current_step_num}/{len(steps_to_run_ordered)}: {step_name.capitalize()} === <<<")

        if script_file in script_args_map:
             stage_args_dict = {k: master_args_dict[k] for k in script_args_map[script_file] if k in master_args_dict}
        else: print(f"[!] Warning: No argument map defined for script: {script_file}", file=sys.stderr)

        # Запускаем скрипт, не ждем после последнего шага
        if not run_stage_script(script_file, python_to_use, stage_args_dict, check=is_critical, wait_for_enter=wait_between_steps and not is_last_step):
             if not is_critical: print(f"[*] Proceeding after non-critical failure in step: {step_name}")
             else: sys.exit(1)

          
    common_utils.clear_VRAM()
    print("\n--- Master Script Finished ---")

# --- Точка входа ---
if __name__ == "__main__":
    main()