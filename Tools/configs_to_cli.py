# Файл: configs_to_cli.py
# -*- coding: utf-8 -*-
import os
import argparse
import toml
import sys
import shlex # Для безопасного форматирования аргументов с пробелами
import glob # Для поиска файлов по шаблону

# Импорт общих утилит
sys.path.append( '../' ) 
try:
    # Ищем common_utils.py в той же директории, что и этот скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr)
    print(f"[-] Please ensure common_utils.py is in the same directory as configs_to_cli.py ({script_dir}).", file=sys.stderr)
    sys.exit(1)
finally:
    if script_dir in sys.path:
        sys.path.remove(script_dir)


# --- Функция загрузки TOML ---
def load_toml_file(filepath):
    """Загружает TOML файл с обработкой ошибок."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        return data
    except FileNotFoundError:
        print(f"[!] Error: Config file not found: {filepath}", file=sys.stderr)
        return None
    except toml.TomlDecodeError as e:
        print(f"[!] Error decoding TOML file {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[!] Unexpected error loading {filepath}: {e}", file=sys.stderr)
        return None

# --- Функция конвертации значения в строку для CLI ---
def format_cli_value(value):
    """Форматирует значение для командной строки."""
    if isinstance(value, bool):
        # Булевы флаги добавляются без значения, если True
        return "" if value else None # Возвращаем пустую строку для флага, None для пропуска
    elif isinstance(value, list):
        # Списки (как optimizer_args) объединяем в строку, экранируя пробелы
        return shlex.join(value)
    elif value is None:
        return None # Пропускаем None значения
    else:
        # Все остальное просто конвертируем в строку
        return str(value)

# --- Основная функция конвертации ---
def convert_configs_to_cli_list(train_config_path, dataset_config_path, project_name, base_dir):
    """Читает конфиги и генерирует список аргументов для master_train.py."""

    print(f"[*] Reading Training Config: {train_config_path}")
    train_data = load_toml_file(train_config_path)
    if not train_data: return None

    print(f"[*] Reading Dataset Config: {dataset_config_path}")
    dataset_data = load_toml_file(dataset_config_path)
    if not dataset_data: return None

    cli_args_list = [] # Список для хранения пар (аргумент, значение)

    # --- Основные параметры ---
    cli_args_list.append(("--project_name", project_name))
    cli_args_list.append(("--base_dir", base_dir))
    cli_args_list.append(("--venv_name", "lora_env")) # Предполагаем дефолт
    cli_args_list.append(("--kohya_dir_name", "kohya_ss")) # Предполагаем дефолт

    # --- Маппинг TOML -> CLI ---
    toml_to_cli_map = {
        # Model Arguments (из training_*.toml)
        "model_arguments.pretrained_model_name_or_path": "base_model",
        "model_arguments.vae": "base_vae",
        "model_arguments.v_parameterization": "v_pred", # store_true

        # Network Arguments (из training_*.toml)
        "network_arguments.unet_lr": "unet_lr",
        "network_arguments.text_encoder_lr": "text_encoder_lr",
        "network_arguments.network_dim": "network_dim",
        "network_arguments.network_alpha": "network_alpha",
        "network_arguments.network_weights": "continue_from_lora",

        # Optimizer Arguments (из training_*.toml)
        "optimizer_arguments.optimizer_type": "optimizer",
        "optimizer_arguments.optimizer_args": "optimizer_args", # Список -> Строка
        "optimizer_arguments.lr_scheduler": "lr_scheduler",
        "optimizer_arguments.lr_scheduler_num_cycles": "lr_scheduler_num_cycles",
        "optimizer_arguments.lr_scheduler_power": "lr_scheduler_power",

        # Dataset Arguments (из training_*.toml)
        "dataset_arguments.cache_latents": "cache_latents", # BooleanOptionalAction
        "dataset_arguments.cache_latents_to_disk": "cache_latents_to_disk", # BooleanOptionalAction
        "dataset_arguments.cache_text_encoder_outputs": "cache_text_encoder_outputs", # store_true
        "dataset_arguments.keep_tokens": "keep_tokens",
        "dataset_arguments.shuffle_caption": "shuffle_tags", # <--- ИЗМЕНЕНИЕ ЗДЕСЬ (BooleanOptionalAction)
        "dataset_arguments.caption_dropout_rate": "caption_dropout",
        "dataset_arguments.caption_dropout_every_n_epochs": "caption_dropout_every_n_epochs",
        "dataset_arguments.caption_tag_dropout_rate": "tag_dropout",
        "dataset_arguments.caption_extension": "caption_extension",

        # Training Arguments (из training_*.toml)
        "training_arguments.save_every_n_epochs": "save_every_n_epochs",
        "training_arguments.save_last_n_epochs": "keep_only_last_n_epochs",
        "training_arguments.max_data_loader_n_workers": "max_data_loader_n_workers",
        "training_arguments.seed": "seed",
        "training_arguments.gradient_checkpointing": "gradient_checkpointing", # BooleanOptionalAction
        "training_arguments.mixed_precision": "precision", # Преобразуется ниже
        "training_arguments.lowram": "lowram", # store_true
        "training_arguments.train_batch_size": "train_batch_size",
        "training_arguments.noise_offset": "noise_offset",
        "training_arguments.min_snr_gamma": "min_snr_gamma",
        "training_arguments.ip_noise_gamma": "ip_noise_gamma",
        "training_arguments.bucket_reso_steps": "bucket_reso_steps",
        "training_arguments.min_bucket_reso": "min_bucket_reso",
        "training_arguments.max_bucket_reso": "max_bucket_reso",
        "training_arguments.bucket_no_upscale": "bucket_no_upscale", # store_true
        "training_arguments.zero_terminal_snr": "zero_terminal_snr", # store_true

        # General (из dataset_*.toml)
        "general.resolution": "resolution",
        "general.flip_aug": "flip_aug", # store_true

        # Datasets.Subsets (из dataset_*.toml, берем первый)
        "datasets.subsets.0.num_repeats": "num_repeats",
    }

    # --- Обработка данных ---
    processed_args = {}

    # 1. Обработка training_config.toml
    for section, params in train_data.items():
        if not isinstance(params, dict): continue
        for key, value in params.items():
            toml_key = f"{section}.{key}"
            cli_key = toml_to_cli_map.get(toml_key)
            if cli_key: processed_args[cli_key] = value

    # 2. Обработка dataset_config.toml
    if 'general' in dataset_data and isinstance(dataset_data['general'], dict):
        for key, value in dataset_data['general'].items():
            toml_key = f"general.{key}"
            cli_key = toml_to_cli_map.get(toml_key)
            if cli_key: processed_args[cli_key] = value

    if 'datasets' in dataset_data and isinstance(dataset_data['datasets'], list) and dataset_data['datasets']:
        if 'subsets' in dataset_data['datasets'][0] and isinstance(dataset_data['datasets'][0]['subsets'], list) and dataset_data['datasets'][0]['subsets']:
            subset = dataset_data['datasets'][0]['subsets'][0]
            for key, value in subset.items():
                 toml_key = f"datasets.subsets.0.{key}"
                 cli_key = toml_to_cli_map.get(toml_key)
                 if cli_key: processed_args[cli_key] = value

    # --- Специальная обработка некоторых аргументов ---
    # preferred_unit и how_many
    max_epochs = train_data.get('training_arguments', {}).get('max_train_epochs')
    max_steps = train_data.get('training_arguments', {}).get('max_train_steps')
    if max_epochs is not None:
        processed_args['preferred-unit'] = 'Epochs'; processed_args['how-many'] = max_epochs
    elif max_steps is not None:
        processed_args['preferred-unit'] = 'Steps'; processed_args['how-many'] = max_steps

    # precision
    if 'precision' in processed_args:
        mp = processed_args['precision']
        full_fp16 = train_data.get('training_arguments', {}).get('full_fp16', False)
        full_bf16 = train_data.get('training_arguments', {}).get('full_bf16', False)
        if mp == 'fp16' and full_fp16: processed_args['precision'] = 'full_fp16'
        elif mp == 'bf16' and full_bf16: processed_args['precision'] = 'full_bf16'
        elif mp == 'no': del processed_args['precision']

    # cross_attention
    sdpa = train_data.get('training_arguments', {}).get('sdpa', False)
    xformers = train_data.get('training_arguments', {}).get('xformers', False)
    if xformers: processed_args['cross-attention'] = 'xformers'
    elif sdpa: processed_args['cross-attention'] = 'sdpa'

    # multinoise
    if train_data.get('training_arguments', {}).get('multires_noise_iterations'):
        processed_args['multinoise'] = True

    # lora_type / conv_*
    network_args = train_data.get('network_arguments', {}).get('network_args')
    if isinstance(network_args, list) and any('conv_dim' in s for s in network_args):
        processed_args['lora-type'] = 'LoCon'
        for arg_str in network_args:
            if 'conv_dim' in arg_str:
                try: processed_args['conv-dim'] = int(arg_str.split('=')[1])
                except: pass
            if 'conv_alpha' in arg_str:
                try: processed_args['conv-alpha'] = int(arg_str.split('=')[1])
                except: pass
    else:
        processed_args['lora-type'] = 'LoRA'

    # --- Формирование финального списка ---
    for key, value in processed_args.items():
        # Используем дефисы для имен аргументов
        cli_arg_name = f"--{key}"
        formatted_value = format_cli_value(value)

        if formatted_value == "" and isinstance(value, bool) and value:
            cli_args_list.append((cli_arg_name, None)) # Булевый флаг
        elif formatted_value is not None:
            cli_args_list.append((cli_arg_name, formatted_value)) # Аргумент со значением

    # Сортируем для лучшей читаемости
    cli_args_list.sort(key=lambda x: x[0])

    return cli_args_list


# --- Парсер аргументов для этого скрипта ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert LoRA training TOML configs back to master_train.py CLI arguments.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("project_name", help="Name of the project (used to find config files).")
    parser.add_argument("--base-dir", default=".", help="Base directory containing the project folder.")
    parser.add_argument("--config-dir", default=None, help="Specify config directory (optional, defaults to <base_dir>/<project_name>/config).")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()

    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    config_dir = args.config_dir if args.config_dir else os.path.join(project_dir, "config")

    print(f"[*] Searching for config files in: {config_dir}")

    # Автоматический поиск файлов
    train_config_pattern = os.path.join(config_dir, f"training_{args.project_name}.toml")
    dataset_config_pattern = os.path.join(config_dir, f"dataset_{args.project_name}.toml")

    train_config_files = glob.glob(train_config_pattern)
    dataset_config_files = glob.glob(dataset_config_pattern)

    if not train_config_files: print(f"[!] Error: Training config file not found: {train_config_pattern}", file=sys.stderr); sys.exit(1)
    if not dataset_config_files: print(f"[!] Error: Dataset config file not found: {dataset_config_pattern}", file=sys.stderr); sys.exit(1)

    train_config_path = train_config_files[0]
    dataset_config_path = dataset_config_files[0]

    print(f"[*] Converting configs for project '{args.project_name}'...")

    cli_args_list = convert_configs_to_cli_list(
        train_config_path,
        dataset_config_path,
        args.project_name,
        args.base_dir
    )

    if cli_args_list:
        print("\n--- Generated master_train.py Arguments (in list format) ---")
        # Выводим в столбик с подчеркиваниями
        for arg_name_cli, arg_value in cli_args_list:
            arg_name_py = arg_name_cli.lstrip('-').replace('-', '_') # Конвертируем для вывода
            if arg_value is None: # Булевый флаг
                print(f"--{arg_name_py}")
            else:
                value_to_print = arg_value
                # Форматирование LR
                if arg_name_cli in ["--unet-lr", "--text-encoder-lr"]:
                    try:
                        float_val = float(arg_value)
                        value_to_print = "{:.0e}".format(float_val).replace("e-0", "e-")
                    except (ValueError, TypeError): pass
                print(f"--{arg_name_py} {shlex.quote(value_to_print)}")
        print("-------------------------------------------------------------")
        print("Note: Paths (--base_model, --base_vae) are taken directly from TOML.")
        print("      You might need to adjust them for your current environment.")
        print("      Step control flags (--run_steps, --skip_steps, --no_wait) must be added manually.")
    else:
        print("\n[!] Failed to generate command line arguments.")
        sys.exit(1)