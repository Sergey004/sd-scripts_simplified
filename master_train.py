# Файл: master_train.py
# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import sys
import platform
import time
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr)
    print(f"[-] Please ensure common_utils.py is in the same directory as master_train.py ({script_dir}).", file=sys.stderr)
    sys.exit(1)
finally:
    if script_dir in sys.path:
        sys.path.remove(script_dir)

# --- Функция для запуска дочернего скрипта с паузой ---
# ... (код run_stage_script без изменений) ...
def run_stage_script(script_name, python_executable, args_dict, check=True, wait_for_enter=True):
    """Запускает указанный скрипт, передавая аргументы из словаря."""
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"[!] Error: Script '{script_name}' not found at {script_path}", file=sys.stderr)
        if check: sys.exit(1)
        return False
    args_list = []
    for key, value in args_dict.items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value: args_list.append(arg_name)
        elif value is not None:
            args_list.extend([arg_name, str(value)])
    command = [python_executable, script_path] + args_list
    print(f"\n--- Running Stage: {script_name} ---")
    result = common_utils.run_cmd(command, check=check)
    success = result is not None and result.returncode == 0
    if success:
        print(f"--- Stage '{script_name}' Finished Successfully ---")
        if wait_for_enter:
             try: input("<?> Press Enter to continue, or Ctrl+C to abort...")
             except KeyboardInterrupt: print("\n[!] Aborted by user."); sys.exit(0)
    else:
        print(f"[!] Stage '{script_name}' failed or finished with errors.", file=sys.stderr)
        if not check:
             try: input("<?> Stage failed (non-critical). Press Enter to continue anyway, or Ctrl+C to abort...")
             except KeyboardInterrupt: print("\n[!] Aborted by user after non-critical failure."); sys.exit(0)
    return success

# --- Парсер аргументов для мастер-скрипта ---
# ... (код parse_master_arguments без изменений) ...
def parse_master_arguments():
    parser = argparse.ArgumentParser(description="Master script to run LoRA training pipeline stages.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Основные Настройки и Пути ---
    g_main = parser.add_argument_group('Main Settings')
    g_main.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    g_main.add_argument("--base_dir", type=str, default=".", help="Base directory for project, kohya_ss, venv.")
    g_main.add_argument("--venv_name", type=str, default="lora_env", help="Name of the venv directory.")
    g_main.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name of the kohya_ss directory.")
    # --- UI/CLI detection ---
    parser.add_argument("--from_ui", action="store_true", help="Flag to indicate launch from Gradio UI.")
    parser.add_argument("--init_project_only", action="store_true", help="Only create project skeleton and exit.")
    # --- Управление Этапами ---
    g_steps = parser.add_argument_group('Pipeline Steps Control')
    g_steps.add_argument("--run_steps", type=str, default="setup,scrape,dedupe,tag,curate,config,train", help="Comma-separated list of steps to run (e.g., 'tag,curate,config,train'). Steps: setup, scrape, dedupe, tag, curate, config, train.")
    g_steps.add_argument("--skip_steps", type=str, default="", help="Comma-separated list of steps to skip (e.g., 'scrape,dedupe'). Overrides --run_steps.")
    g_steps.add_argument("--no_wait", action='store_true', help="Do not wait for Enter key between steps.")
    g_steps.add_argument("--skip_initial_pause", action='store_true', help="Skip the initial pause for dataset preparation.")
    # --- Собираем ВСЕ аргументы для ВСЕХ этапов ---
    g_scrape = parser.add_argument_group('Step 1: Scrape Images Args')
    g_scrape.add_argument("--scrape_tags", type=str, default="", help="Tags for Gelbooru scraping.")
    g_scrape.add_argument("--scrape_limit", type=int, default=1000, help="Max images to fetch.")
    g_scrape.add_argument("--scrape_max_res", type=int, default=3072, help="Max image resolution for scraping.")
    g_scrape.add_argument("--scrape_include_parents", action=argparse.BooleanOptionalAction, default=True, help="Include Gelbooru posts with parents.")
    g_dedupe = parser.add_argument_group('Step 2: Detect Duplicates Args')
    g_dedupe.add_argument("--dedup_threshold", type=float, default=0.985, help="Similarity threshold for duplicates.")
    g_tag = parser.add_argument_group('Step 3: Tag Images Args')
    g_tag.add_argument("--tagging_method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    g_tag.add_argument("--tagger_threshold", type=float, default=0.35, help="Confidence threshold for WD14.")
    g_tag.add_argument("--tagger_batch_size", type=int, default=8, help="Batch size for WD14.")
    g_tag.add_argument("--blip_min_length", type=int, default=10, help="Min caption length for BLIP.")
    g_tag.add_argument("--blip_max_length", type=int, default=75, help="Max caption length for BLIP.")
    g_tag.add_argument("--tagger_blacklist", type=str, default="bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber", help="Tags to blacklist after WD14 tagging.")
    g_tag.add_argument("--overwrite_tags", action='store_true', help="Overwrite existing tag files during tagging.")
    g_curate = parser.add_argument_group('Step 4: Curate Tags Args')
    g_curate.add_argument("--caption_extension", type=str, default=".txt", help="Extension for tag/caption files.")
    g_curate.add_argument("--activation_tag", type=str, default="", help="Activation tag(s) to prepend.")
    g_curate.add_argument("--remove_tags", type=str, default="lowres, bad anatomy, worst quality, low quality", help="Tags to remove during curation.")
    g_curate.add_argument("--search_tags", type=str, default="", help="Tags to search for replacement.")
    g_curate.add_argument("--replace_tags", type=str, default="", help="Tags to replace with.")
    g_curate.add_argument("--sort_tags_alpha", action='store_true', help="Sort tags alphabetically.")
    g_curate.add_argument("--remove_duplicate_tags", action='store_true', help="Remove duplicate tags.")
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
    g_config.add_argument("--caption_dropout", type=float, default=0.0, metavar='RATE', help="Caption dropout rate (0-1).")
    g_config.add_argument("--caption_dropout_every_n_epochs", type=int, default=0, metavar='N', help="Apply caption dropout every N epochs (0=never).")
    g_config.add_argument("--tag_dropout", type=float, default=0.0, metavar='RATE', help="Tag dropout rate (0-1).")
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
    g_run_train = parser.add_argument_group('Step 6: Run Training Args')
    g_run_train.add_argument("--num_cpu_threads", type=int, default=2, help="CPU threads per process for Accelerate.")

    return parser.parse_args()

# --- Основная Логика ---
def main():
    master_args = parse_master_arguments()
    master_args_dict = vars(master_args).copy()

    # === UI/CLI detection logic ===
    launched_from_ui = getattr(master_args, "from_ui", False)
    if launched_from_ui:
        print("[INFO] Launched from Gradio UI (--from_ui detected)")
    else:
        print("[INFO] Launched from CLI (no --from_ui)")

    # --- Если только создание скелета проекта ---
    if getattr(master_args, "init_project_only", False):
        base_dir = os.path.abspath(master_args.base_dir)
        project_dir = os.path.join(base_dir, master_args.project_name)
        paths = {
            "project": project_dir,
            "images": os.path.join(project_dir, "dataset"),
            "output": os.path.join(project_dir, "output"),
            "logs": os.path.join(project_dir, "logs"),
            "config": os.path.join(project_dir, "config"),
        }
        print("[*] Creating project skeleton...")
        for key in ["project", "images", "output", "logs", "config"]:
            try:
                os.makedirs(paths[key], exist_ok=True)
                print(f"[+] Created: {paths[key]}")
            except OSError as e:
                print(f"[X] CRITICAL ERROR: Could not create directory {paths[key]}: {e}", file=sys.stderr)
                sys.exit(1)
        print("[✓] Project skeleton created.")
        return

    # --- Отключаем паузы при запуске из UI ---
    wait_between_steps = not master_args.no_wait
    skip_initial_pause = master_args.skip_initial_pause
    if launched_from_ui:
        wait_between_steps = False
        skip_initial_pause = True
        print("[INFO] All pauses are disabled for UI mode.")

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

    # wait_between_steps и skip_initial_pause уже определены выше
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
        print("-" * 20); print(f"<?> Please prepare your dataset now."); print(f"    Ensure images are inside: {paths['images']}")
        if "tag" not in steps_to_run and "curate" not in steps_to_run: print(f"    Ensure tag files ({master_args.caption_extension}) are также in that folder.")
        try: input("<?> Press Enter when ready, or Ctrl+C to abort...")
        except KeyboardInterrupt: print("\n[!] Aborted by user."); sys.exit(0)
        print("-" * 20)
    else: print("[*] Skipping initial dataset preparation pause.")

    # Проверка на пустой датасет
    steps_requiring_images = {"tag", "curate", "config", "train"}
    if not steps_to_run.isdisjoint(steps_requiring_images):
        image_count = common_utils.get_image_count(paths['images'])
        if image_count == 0: print(f"[X] CRITICAL ERROR: Dataset folder '{paths['images']}' is empty.", file=sys.stderr); sys.exit(1)

    # --- Авто-определение параметров ---
    # Делаем это *после* возможной паузы и *до* запуска шагов,
    # чтобы использовать актуальное состояние (например, кол-во картинок)
    final_params = {}
    gpu_vram = common_utils.get_gpu_vram()
    auto_vram_settings = {}
    if master_args.auto_vram_params: auto_vram_settings = common_utils.determine_vram_parameters(gpu_vram); print(f"[*] Auto VRAM params: {auto_vram_settings}")

    final_params['train_batch_size'] = master_args.train_batch_size if master_args.train_batch_size is not None else auto_vram_settings.get('batch_size', 1)
    final_params['optimizer'] = master_args.optimizer if master_args.optimizer is not None else auto_vram_settings.get('optimizer', 'AdamW8bit')
    final_params['precision'] = master_args.precision if master_args.precision is not None else auto_vram_settings.get('precision', 'fp16')
    final_params['optimizer_args'] = [] # Важно инициализировать
    if master_args.use_recommended_optimizer_args:
        optimizer_lower = final_params['optimizer'].lower()
        if optimizer_lower == "adamw8bit": final_params['optimizer_args'] = ["weight_decay=0.1", "betas=[0.9,0.99]"]
        elif optimizer_lower == "prodigy": final_params['optimizer_args'] = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
        # ... другие рекомендованные ...
    elif master_args.optimizer_args:
         # Разделяем строку аргументов, переданную пользователем
         final_params['optimizer_args'] = master_args.optimizer_args.split()

    image_count = common_utils.get_image_count(paths['images']) # Получаем актуальное кол-во
    auto_repeats_val = 0
    if master_args.auto_repeats: auto_repeats_val = common_utils.determine_num_repeats(image_count); print(f"[*] Auto num_repeats: {auto_repeats_val} (based on {image_count} images)")
    final_params['num_repeats'] = master_args.num_repeats if master_args.num_repeats is not None else auto_repeats_val if master_args.auto_repeats else 10

    print("--- Final Effective Parameters ---")
    print(f"  Batch Size: {final_params['train_batch_size']}"); print(f"  Optimizer: {final_params['optimizer']}"); print(f"  Optimizer Args: {' '.join(final_params['optimizer_args']) if final_params['optimizer_args'] else 'None'}"); print(f"  Precision: {final_params['precision']}"); print(f"  Num Repeats: {final_params['num_repeats']}"); print("--------------------------------")

    # ===> ОБНОВЛЯЕМ master_args_dict ФИНАЛЬНЫМИ ЗНАЧЕНИЯМИ <===
    for key, value in final_params.items():
        # Особая обработка списка optimizer_args для передачи в командную строку
        if key == 'optimizer_args':
            master_args_dict[key] = ' '.join(value) if value else ""
        else:
            master_args_dict[key] = value
    # =========================================================


    # --- Определяем аргументы для каждого скрипта ---
    script_args_map = {
        "0_setup_environment.py": ['base_dir', 'venv_name', 'kohya_dir_name'],
        "1_scrape_images.py": ['project_name', 'base_dir', 'scrape_tags', 'scrape_limit', 'scrape_max_res', 'scrape_include_parents'],
        "2_detect_duplicates.py": ['project_name', 'base_dir', 'dedup_threshold'],
        "3_tag_images.py": ['project_name', 'base_dir', 'kohya_dir_name', 'venv_name', 'tagging_method', 'tagger_threshold', 'tagger_batch_size', 'blip_min_length', 'blip_max_length', 'caption_extension', 'tagger_blacklist', 'overwrite_tags'],
        "4_curate_tags.py": ['project_name', 'base_dir', 'caption_extension', 'activation_tag', 'remove_tags', 'search_tags', 'replace_tags', 'sort_tags_alpha', 'remove_duplicate_tags'],
        "5_generate_configs.py": [
            'project_name', 'base_dir', 'base_model', 'custom_model', 'base_vae', 'custom_vae', 'v_pred',
            'resolution', 'shuffle_tags', 'keep_tokens', 'flip_aug',
            'num_repeats', 'auto_repeats', 'preferred_unit', 'how_many', 'save_every_n_epochs', 'keep_only_last_n_epochs',
            'caption_dropout', 'caption_dropout_every_n_epochs', 'tag_dropout',
            'unet_lr', 'text_encoder_lr', 'lr_scheduler', 'lr_scheduler_num_cycles',
            'lr_scheduler_power', 'lr_warmup_ratio', 'min_snr_gamma', 'noise_offset', 'ip_noise_gamma', 'multinoise',
            'zero_terminal_snr', 'lora_type', 'network_dim', 'network_alpha', 'conv_dim', 'conv_alpha', 'continue_from_lora',
            'auto_vram_params', 'train_batch_size', 'cross_attention', 'precision', 'cache_latents', 'cache_latents_to_disk',
            'cache_text_encoder_outputs', 'gradient_checkpointing', 'optimizer',
            # ===> ДОБАВЛЕН КЛЮЧ <===
            'use_recommended_optimizer_args',
            # ===> КОНЕЦ ДОБАВЛЕНИЯ <===
            'optimizer_args', 'max_data_loader_n_workers', 'seed', 'lowram', 'bucket_reso_steps', 'min_bucket_reso',
            'max_bucket_reso', 'bucket_no_upscale', 'caption_extension'
        ],
        "6_run_training.py": ['project_name', 'base_dir', 'kohya_dir_name', 'venv_name', 'num_cpu_threads']
    }

    # --- Запуск этапов ---
    current_step_num = 0
    steps_to_run_ordered = [s for s in all_steps if s in steps_to_run]

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
             # Используем обновленный master_args_dict для получения значений
             stage_args_dict = {k: master_args_dict[k] for k in script_args_map[script_file] if k in master_args_dict}
        else: print(f"[!] Warning: No argument map defined for script: {script_file}", file=sys.stderr); stage_args_dict = {}

        if not run_stage_script(script_file, python_to_use, stage_args_dict, check=is_critical, wait_for_enter=wait_between_steps and not is_last_step):
             if not is_critical: print(f"[*] Proceeding after non-critical failure in step: {step_name}")
             else: sys.exit(1)

    # Очистка VRAM после завершения всех шагов
    common_utils.clear_VRAM()
    print("\n--- Master Script Finished ---")

# --- Точка входа ---
if __name__ == "__main__":
    main()