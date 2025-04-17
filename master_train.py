# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import sys
import platform

# --- Утилита для запуска команд (Копия) ---
def run_cmd(command, check=True, shell=False, capture_output=False, text=False, cwd=None, env=None):
    """Утилита для запуска команд оболочки с улучшенной обработкой ошибок."""
    command_str = ' '.join(command) if isinstance(command, list) else command
    print(f"[*] Running: {command_str}" + (f" in {cwd}" if cwd else ""))
    try:
        process = subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text, cwd=cwd, env=env, encoding='utf-8', errors='ignore')
        if capture_output:
             stdout = process.stdout.strip() if process.stdout else ""
             stderr = process.stderr.strip() if process.stderr else ""
             if stdout: print(f"  [stdout]:\n{stdout}")
             if stderr: print(f"  [stderr]:\n{stderr}", file=sys.stderr)
        if process.returncode != 0:
             print(f"[!] Warning: Command '{command_str}' finished with non-zero exit code: {process.returncode}", file=sys.stderr)
             if check:
                  print(f"[X] Critical command failed. Exiting.")
                  sys.exit(1)
        return process
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running command: {command_str}", file=sys.stderr); print(f"[!] Exit code: {e.returncode}", file=sys.stderr);
        if capture_output:
            stdout = e.stdout.decode(errors='ignore').strip() if isinstance(e.stdout, bytes) else (e.stdout.strip() if e.stdout else ""); stderr = e.stderr.decode(errors='ignore').strip() if isinstance(e.stderr, bytes) else (e.stderr.strip() if e.stderr else "")
            if stdout: print(f"[!] Stdout:\n{stdout}", file=sys.stderr);
            if stderr: print(f"[!] Stderr:\n{stderr}", file=sys.stderr);
        print(f"[X] Critical command failed. Exiting.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[!] Error: Command not found - {command[0]}. Is it installed and in PATH?", file=sys.stderr); print(f"[!] Details: {e}", file=sys.stderr);
        if check: print(f"[X] Critical command not found. Exiting."); sys.exit(1);
        return None
    except Exception as e:
        print(f"[!] An unexpected error occurred running command '{command_str}': {e}", file=sys.stderr);
        import traceback; traceback.print_exc();
        if check: print(f"[X] Unexpected error during critical command. Exiting."); sys.exit(1);
        return None

# --- Функция для запуска дочернего скрипта с паузой ---
def run_stage_script(script_name, python_executable, args_list, check=True, wait_for_enter=True):
    """Запускает указанный скрипт с заданными аргументами и опциональной паузой."""
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"[!] Error: Script '{script_name}' not found at {script_path}", file=sys.stderr)
        if check: sys.exit(1)
        return False

    command = [python_executable, script_path] + args_list
    print(f"\n--- Running Stage: {script_name} ---")
    result = run_cmd(command, check=check)

    success = result is not None and result.returncode == 0

    if success:
        print(f"--- Stage '{script_name}' Finished Successfully ---")
        if wait_for_enter:
             try:
                 input("<?> Press Enter to continue to the next step, or Ctrl+C to abort...")
             except KeyboardInterrupt:
                 print("\n[!] Aborted by user.")
                 sys.exit(0)
    else:
        print(f"[!] Stage '{script_name}' failed or finished with errors.", file=sys.stderr)
        if check:
             # check=True уже вызвал бы sys.exit(1) в run_cmd
             pass
        else:
             # Если ошибка не критична, все равно спросим пользователя
             try:
                 input("<?> Stage failed (non-critical). Press Enter to continue anyway, or Ctrl+C to abort...")
             except KeyboardInterrupt:
                 print("\n[!] Aborted by user after non-critical failure.")
                 sys.exit(0)

    return success


# --- Парсер аргументов (без изменений) ---
def parse_master_arguments():
    parser = argparse.ArgumentParser(description="Master script to run LoRA training pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ... (весь код парсера аргументов из предыдущего ответа) ...
    # --- Основные Настройки и Пути ---
    g_main = parser.add_argument_group('Main Settings')
    g_main.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    g_main.add_argument("--base_dir", type=str, default=".", help="Base directory containing project folders, kohya_ss, and venv.")
    g_main.add_argument("--venv_name", type=str, default="lora_env", help="Name of the Python virtual environment directory.")
    g_main.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name of the kohya_ss scripts directory.")
    # --- Управление Этапами ---
    g_steps = parser.add_argument_group('Pipeline Steps Control')
    g_steps.add_argument("--skip_setup", action='store_true', help="Skip environment setup (Step 0).")
    g_steps.add_argument("--skip_scrape", action='store_true', help="Skip image scraping (Step 1).")
    g_steps.add_argument("--skip_deduplication", action='store_true', help="Skip duplicate detection (Step 2).")
    g_steps.add_argument("--skip_tagging", action='store_true', help="Skip automatic tagging (Step 3).")
    g_steps.add_argument("--skip_curation", action='store_true', help="Skip tag curation (Step 4).")
    g_steps.add_argument("--skip_config", action='store_true', help="Skip config generation (Step 5).")
    g_steps.add_argument("--skip_training", action='store_true', help="Skip the actual training (Step 6).")
    g_steps.add_argument("--run_prep_only", action='store_true', help="Run only preparation steps (1-5) and stop.")
    g_steps.add_argument("--run_train_only", action='store_true', help="Run only the training step (6).")
    g_steps.add_argument("--no_wait", action='store_true', help="Do not wait for Enter key between steps.") # Новый флаг для отключения паузы

    # --- Добавляем ВСЕ аргументы из дочерних скриптов ---
    # Аргументы из 1_scrape_images.py
    g_scrape = parser.add_argument_group('Step 1: Scrape Images Args')
    g_scrape.add_argument("--scrape_tags", type=str, default="", help="Tags for Gelbooru scraping.")
    g_scrape.add_argument("--scrape_limit", type=int, default=1000, help="Max images to fetch.")
    g_scrape.add_argument("--scrape_max_res", type=int, default=3072, help="Max image resolution for scraping.")
    g_scrape.add_argument("--scrape_include_parents", action=argparse.BooleanOptionalAction, default=True, help="Include Gelbooru posts with parents.")
    # Аргументы из 2_detect_duplicates.py
    g_dedupe = parser.add_argument_group('Step 2: Detect Duplicates Args')
    g_dedupe.add_argument("--dedup_threshold", type=float, default=0.985, help="Similarity threshold for duplicates.")
    # Аргументы из 3_tag_images.py
    g_tag = parser.add_argument_group('Step 3: Tag Images Args')
    g_tag.add_argument("--tagging_method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    g_tag.add_argument("--tagger_threshold", type=float, default=0.35, help="Confidence threshold for WD14.")
    g_tag.add_argument("--tagger_batch_size", type=int, default=8, help="Batch size for WD14.")
    g_tag.add_argument("--blip_min_length", type=int, default=10, help="Min caption length for BLIP.")
    g_tag.add_argument("--blip_max_length", type=int, default=75, help="Max caption length for BLIP.")
    g_tag.add_argument("--tagger_blacklist", type=str, default="bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber", help="Tags to blacklist after WD14 tagging.")
    g_tag.add_argument("--overwrite_tags", action='store_true', help="Overwrite existing tag files during tagging.")
    # Аргументы из 4_curate_tags.py
    g_curate = parser.add_argument_group('Step 4: Curate Tags Args')
    g_curate.add_argument("--caption_extension", type=str, default=".txt", help="Extension for tag/caption files.")
    g_curate.add_argument("--activation_tag", type=str, default="", help="Activation tag(s) to prepend.")
    g_curate.add_argument("--remove_tags", type=str, default="lowres, bad anatomy, worst quality, low quality", help="Tags to remove during curation.")
    g_curate.add_argument("--search_tags", type=str, default="", help="Tags to search for replacement.")
    g_curate.add_argument("--replace_tags", type=str, default="", help="Tags to replace with.")
    g_curate.add_argument("--sort_tags_alpha", action='store_true', help="Sort tags alphabetically.")
    g_curate.add_argument("--remove_duplicate_tags", action='store_true', help="Remove duplicate tags.")
    # Аргументы из 5_generate_configs.py (используются также в 6_run_training.py через конфиг)
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

    # Определяем пути
    base_dir = os.path.abspath(master_args.base_dir)
    venv_dir = os.path.join(base_dir, master_args.venv_name)
    kohya_dir = os.path.join(base_dir, master_args.kohya_dir_name)
    project_dir = os.path.join(base_dir, master_args.project_name)

    # Определяем путь к Python внутри venv
    venv_python = os.path.join(venv_dir, 'bin', 'python') if platform.system() != 'Windows' else os.path.join(venv_dir, 'Scripts', 'python.exe')

    # Определяем, какие шаги выполнять
    # Режимы prep_only и train_only имеют приоритет
    if master_args.run_prep_only:
        run_setup = not master_args.skip_setup
        run_scrape = not master_args.skip_scrape
        run_dedupe = not master_args.skip_deduplication
        run_tag = not master_args.skip_tagging
        run_curate = not master_args.skip_curation
        run_config = not master_args.skip_config
        run_training = False # Главное отличие prep_only
    elif master_args.run_train_only:
        run_setup = False # Setup не нужен для train_only
        run_scrape = False
        run_dedupe = False
        run_tag = False
        run_curate = False
        run_config = False # Конфиги должны уже быть
        run_training = True # Главное отличие train_only
    else: # Режим по умолчанию или --run_all
        run_setup = not master_args.skip_setup
        run_scrape = not master_args.skip_scrape
        run_dedupe = not master_args.skip_deduplication
        run_tag = not master_args.skip_tagging
        run_curate = not master_args.skip_curation
        run_config = not master_args.skip_config
        run_training = not master_args.skip_training

    wait_between_steps = not master_args.no_wait # Ждать, если флаг --no_wait не указан

    print("--- Master Script Start ---")
    # ... (печать информации о проекте и шагах) ...
    print(f"Project: {master_args.project_name}")
    print(f"Base Dir: {base_dir}")
    print(f"Wait between steps: {wait_between_steps}")
    print("-" * 20)
    print("Running Steps:")
    print(f"  Setup Env: {run_setup}")
    print(f"  Scrape:    {run_scrape}")
    print(f"  Dedupe:    {run_dedupe}")
    print(f"  Tag:       {run_tag}")
    print(f"  Curate:    {run_curate}")
    print(f"  Config:    {run_config}")
    print(f"  Training:  {run_training}")
    print("-" * 20)

    # --- Шаг 0: Настройка Окружения ---
    if run_setup:
        setup_args = [ "--base_dir", base_dir, "--venv_name", master_args.venv_name, "--kohya_dir_name", master_args.kohya_dir_name ]
        # Запускаем системным Python, ждем завершения, check=True (критично)
        if not run_stage_script("setup_environment.py", sys.executable, setup_args, check=True, wait_for_enter=wait_between_steps):
             sys.exit(1) # Выход, если setup не удался
    else:
        print("[*] Skipping Step 0: Environment Setup.")
        # Проверяем наличие venv python, если пропускаем setup
        if not os.path.exists(venv_python):
             print(f"[X] CRITICAL ERROR: Venv Python not found at {venv_python}, but setup was skipped!", file=sys.stderr)
             print("[-] Run without --skip_setup or run setup_environment.py manually first.")
             sys.exit(1)


    # --- Шаг 1: Скачивание ---
    if run_scrape:
        scrape_args = [ "--project_name", master_args.project_name, "--base_dir", base_dir, "--scrape_tags", master_args.scrape_tags, "--scrape_limit", str(master_args.scrape_limit), "--scrape_max_res", str(master_args.scrape_max_res) ]
        if not master_args.scrape_include_parents: scrape_args.append("--no-scrape_include_parents")
        run_stage_script("1_scrape_images.py", venv_python, scrape_args, check=False, wait_for_enter=wait_between_steps) # check=False
    elif not master_args.run_train_only: # Печатаем пропуск только если не train_only
        print("[*] Skipping Step 1: Scrape Images.")


    # --- Шаг 2: Дедупликация ---
    if run_dedupe:
        dedupe_args = [ "--project_name", master_args.project_name, "--base_dir", base_dir, "--dedup_threshold", str(master_args.dedup_threshold) ]
        run_stage_script("2_detect_duplicates.py", venv_python, dedupe_args, check=False, wait_for_enter=wait_between_steps) # check=False
    elif not master_args.run_train_only:
        print("[*] Skipping Step 2: Detect Duplicates.")


    # --- Шаг 3: Тегирование ---
    if run_tag:
        tag_args = [ "--project_name", master_args.project_name, "--base_dir", base_dir, "--kohya_dir_name", master_args.kohya_dir_name, "--venv_name", master_args.venv_name, "--tagging_method", master_args.tagging_method, "--tagger_threshold", str(master_args.tagger_threshold), "--tagger_batch_size", str(master_args.tagger_batch_size), "--blip_min_length", str(master_args.blip_min_length), "--blip_max_length", str(master_args.blip_max_length), "--caption_extension", master_args.caption_extension, "--tagger_blacklist", master_args.tagger_blacklist ]
        if master_args.overwrite_tags: tag_args.append("--overwrite_tags")
        # Ошибка тегирования может быть критичной для следующих шагов
        if not run_stage_script("3_tag_images.py", venv_python, tag_args, check=True, wait_for_enter=wait_between_steps):
             sys.exit(1)
    elif not master_args.run_train_only:
        print("[*] Skipping Step 3: Tag Images.")


    # --- Шаг 4: Курирование тегов ---
    if run_curate:
        curate_args = [ "--project_name", master_args.project_name, "--base_dir", base_dir, "--caption_extension", master_args.caption_extension, "--activation_tag", master_args.activation_tag, "--remove_tags", master_args.remove_tags, "--search_tags", master_args.search_tags, "--replace_tags", master_args.replace_tags ]
        if master_args.sort_tags_alpha: curate_args.append("--sort_tags_alpha")
        if master_args.remove_duplicate_tags: curate_args.append("--remove_duplicate_tags")
        # Ошибка курирования может быть критичной
        if not run_stage_script("4_curate_tags.py", venv_python, curate_args, check=True, wait_for_enter=wait_between_steps):
             sys.exit(1)
    elif not master_args.run_train_only:
        print("[*] Skipping Step 4: Curate Tags.")


    # --- Шаг 5: Генерация Конфигов ---
    if run_config:
        # Собираем все аргументы для 5_generate_configs.py
        config_gen_args_dict = vars(master_args).copy() # Копируем все аргументы
        # Аргументы, которые не нужны скрипту 5
        args_to_remove_for_config = {
            'skip_setup', 'skip_scrape', 'skip_deduplication', 'skip_tagging',
            'skip_curation', 'skip_config', 'skip_training', 'run_prep_only',
            'run_train_only', 'no_wait', 'scrape_tags', 'scrape_limit',
            'scrape_max_res', 'scrape_include_parents', 'dedup_threshold',
            'tagging_method', 'tagger_threshold', 'tagger_batch_size',
            'blip_min_length', 'blip_max_length', 'tagger_blacklist',
            'overwrite_tags', 'activation_tag', 'remove_tags', 'search_tags',
            'replace_tags', 'sort_tags_alpha', 'remove_duplicate_tags',
            'num_cpu_threads', 'kohya_dir_name', 'venv_name' # Добавил venv_name
        }
        config_gen_args = []
        print("[*] Preparing arguments for 5_generate_configs.py...")
        for key, value in config_gen_args_dict.items():
            if key not in args_to_remove_for_config:
                arg_name = f"--{key.replace('_', '-')}"
                # Обработка булевых флагов (action='store_true' или BooleanOptionalAction)
                if isinstance(value, bool):
                    # Если значение True, добавляем флаг
                    if value:
                        config_gen_args.append(arg_name)
                    # Если значение False и это BooleanOptionalAction (имеет --no- версию),
                    # то ничего не добавляем (отсутствие флага = False)
                    # Если это store_true и значение False, тоже ничего не добавляем.
                # Обработка остальных типов (строки, числа и т.д.)
                elif value is not None:
                    config_gen_args.extend([arg_name, str(value)])
                # Игнорируем аргументы со значением None

        # print(f"  Args for config script: {config_gen_args}") # Для дебага

        # Генерация конфигов критична для тренировки
        if not run_stage_script("5_generate_configs.py", venv_python, config_gen_args, check=True, wait_for_enter=wait_between_steps):
            sys.exit(1)
    elif not master_args.run_train_only:
        print("[*] Skipping Step 5: Generate Configs.")


    # --- Шаг 6: Тренировка ---
    if run_training:
        train_args = [ "--project_name", master_args.project_name, "--base_dir", base_dir, "--kohya_dir_name", master_args.kohya_dir_name, "--venv_name", master_args.venv_name, "--num_cpu_threads", str(master_args.num_cpu_threads) ]
        # Запускаем тренировку, не ждем Enter после нее
        if not run_stage_script("6_run_training.py", venv_python, train_args, check=True, wait_for_enter=False):
             sys.exit(1) # Выходим с ошибкой, если тренировка не удалась
    else:
        print("[*] Skipping Step 6: Run Training.")


    print("\n--- Master Script Finished ---")

# --- Точка входа ---
if __name__ == "__main__":
    main()