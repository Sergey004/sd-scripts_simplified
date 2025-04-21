# Файл: 5_generate_configs.py
# -*- coding: utf-8 -*-
import os
import argparse
import toml
import math
import sys
import re
# Импорт общих утилит
try:
    # Ищем common_utils.py в той же директории
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr); sys.exit(1)
finally:
    if script_dir in sys.path: sys.path.remove(script_dir)


# --- Функция генерации конфигов ---
def generate_configuration_files(paths, args, final_params):
    """Генерирует training_*.toml и dataset_*.toml."""
    print("\n--- Generating Configuration Files ---")

    current_num_repeats = final_params['num_repeats']; current_batch_size = final_params['train_batch_size']
    current_optimizer = final_params['optimizer']; current_precision = final_params['precision']
    current_optimizer_args = final_params['optimizer_args']

    images_folder = paths["images"]; config_folder = paths["config"]
    output_folder = paths["output"]; log_folder = paths["logs"]

    # Пересчитываем шаги/эпохи
    total_images = common_utils.get_image_count(images_folder)
    if total_images == 0: print(f"[!] Error: No images found in {images_folder}.", file=sys.stderr); return None
    if current_batch_size <= 0: print("[!] Error: Batch size must be positive.", file=sys.stderr); return None
    if current_num_repeats <= 0: print("[!] Error: Repeats must be positive.", file=sys.stderr); return None

    pre_steps_per_epoch = total_images * current_num_repeats
    steps_per_epoch = math.ceil(pre_steps_per_epoch / current_batch_size)

    max_train_epochs = None; max_train_steps = None
    if args.preferred_unit == "Epochs": max_train_epochs = args.how_many; max_train_steps = max_train_epochs * steps_per_epoch
    else: max_train_steps = args.how_many; max_train_epochs = math.ceil(max_train_steps / steps_per_epoch) if steps_per_epoch > 0 else 0

    if max_train_steps <= 0: print(f"[!] Error: Calculated total steps ({max_train_steps}) <= 0.", file=sys.stderr); return None
    lr_warmup_steps = int(max_train_steps * args.lr_warmup_ratio) if args.lr_scheduler not in ('constant', 'cosine') else 0

    # Определение путей модели/VAE
    model_url = args.custom_model if args.custom_model else args.base_model
    vae_url = args.custom_vae if args.custom_vae else args.base_vae
    model_file_path = os.path.abspath(model_url) if os.path.exists(model_url) else model_url
    vae_file_path = os.path.abspath(vae_url) if vae_url and os.path.exists(vae_url) else vae_url
    is_diffusers_model = os.path.isdir(model_file_path) if os.path.exists(model_file_path) else ('huggingface.co/' in model_file_path and not model_file_path.endswith(('.safetensors', '.ckpt')))

    print(f"[*] Using Model Path/ID: {model_file_path}" + (" (Diffusers)" if is_diffusers_model else ""))
    if vae_file_path: print(f"[*] Using VAE Path/ID: {vae_file_path}")
    else: print("[*] No external VAE specified.")

    # Генерация конфигов
    os.makedirs(config_folder, exist_ok=True)
    config_file = os.path.join(config_folder, f"training_{args.project_name}.toml")
    dataset_config_file = os.path.join(config_folder, f"dataset_{args.project_name}.toml")

    network_args_list = []
    if args.lora_type.lower() == "locon": network_args_list = [f"conv_dim={args.conv_dim}", f"conv_alpha={args.conv_alpha}"]

    mixed_precision_val = "no"; full_precision_val = False
    if "fp16" in current_precision: mixed_precision_val = "fp16"
    if "bf16" in current_precision: mixed_precision_val = "bf16"
    if "full" in current_precision: full_precision_val = True

    # Собираем training_dict
    training_dict = {
        "model_arguments": {"pretrained_model_name_or_path": model_file_path, "vae": vae_file_path if vae_file_path and not is_diffusers_model else None, "v_parameterization": args.v_pred},
        "network_arguments": {"unet_lr": args.unet_lr, "text_encoder_lr": args.text_encoder_lr if not args.cache_text_encoder_outputs else 0, "network_dim": args.network_dim, "network_alpha": args.network_alpha, "network_module": "networks.lora", "network_args": network_args_list or None, "network_train_unet_only": args.text_encoder_lr == 0 or args.cache_text_encoder_outputs, "network_weights": args.continue_from_lora},
        "optimizer_arguments": {"optimizer_type": current_optimizer, "learning_rate": args.unet_lr, "optimizer_args": current_optimizer_args or None, "lr_scheduler": args.lr_scheduler, "lr_warmup_steps": lr_warmup_steps, "lr_scheduler_num_cycles": args.lr_scheduler_num_cycles if args.lr_scheduler == "cosine_with_restarts" else None, "lr_scheduler_power": args.lr_scheduler_power if args.lr_scheduler == "polynomial" else None, "max_grad_norm": 1.0, "loss_type": "l2"},
        "dataset_arguments": {
            "cache_latents": args.cache_latents,
            "cache_latents_to_disk": args.cache_latents_to_disk,
            "cache_text_encoder_outputs": args.cache_text_encoder_outputs,
            "keep_tokens": args.keep_tokens,
            "shuffle_caption": args.shuffle_tags and not args.cache_text_encoder_outputs,
            # ===> ИСПОЛЬЗУЕМ НОВЫЕ АРГУМЕНТЫ <===
            "caption_dropout_rate": args.caption_dropout or None,
            "caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs or None,
            # ===> КОНЕЦ ИЗМЕНЕНИЙ <===
            "caption_tag_dropout_rate": args.tag_dropout or None,
            "caption_extension": args.caption_extension
        },
        "training_arguments": {"output_dir": output_folder, "output_name": args.project_name, "save_precision": "fp16", "save_every_n_epochs": args.save_every_n_epochs or None, "save_last_n_epochs": args.keep_only_last_n_epochs or None, "save_model_as": "safetensors", "max_train_epochs": max_train_epochs if args.preferred_unit == "Epochs" else None, "max_train_steps": max_train_steps if args.preferred_unit == "Steps" else None, "max_data_loader_n_workers": args.max_data_loader_n_workers, "persistent_data_loader_workers": True, "seed": args.seed, "gradient_checkpointing": args.gradient_checkpointing, "gradient_accumulation_steps": 1, "mixed_precision": mixed_precision_val, "full_fp16": full_precision_val if mixed_precision_val == "fp16" else None, "full_bf16": full_precision_val if mixed_precision_val == "bf16" else None, "logging_dir": log_folder, "log_prefix": args.project_name, "log_with": "tensorboard", "lowram": args.lowram, "train_batch_size": current_batch_size, "xformers": args.cross_attention == "xformers", "sdpa": args.cross_attention == "sdpa", "noise_offset": args.noise_offset or None, "min_snr_gamma": args.min_snr_gamma if args.min_snr_gamma > 0 else None, "ip_noise_gamma": args.ip_noise_gamma if args.ip_noise_gamma > 0 else None, "multires_noise_iterations": 6 if args.multinoise else None, "multires_noise_discount": 0.3 if args.multinoise else None, "max_token_length": 225, "bucket_reso_steps": args.bucket_reso_steps, "min_bucket_reso": args.min_bucket_reso, "max_bucket_reso": args.max_bucket_reso, "bucket_no_upscale": args.bucket_no_upscale, "enable_bucket": True, "zero_terminal_snr": args.zero_terminal_snr}
    }

    def remove_none_recursive(d):
        if isinstance(d, dict): return {k: remove_none_recursive(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list): return [remove_none_recursive(i) for i in d] # Не удаляем None из списков аргументов
        else: return d

    clean_training_dict = remove_none_recursive(training_dict)
    try:
        with open(config_file, "w", encoding='utf-8') as f: toml.dump(clean_training_dict, f)
        print(f"  Training config saved: {config_file}")
    except Exception as e: print(f"[!] Error writing training config {config_file}: {e}", file=sys.stderr); return None

    dataset_dict = {
      "general": {"resolution": args.resolution, "keep_tokens": args.keep_tokens, "flip_aug": args.flip_aug, "enable_bucket": True, "bucket_reso_steps": args.bucket_reso_steps, "min_bucket_reso": args.min_bucket_reso, "max_bucket_reso": args.max_bucket_reso, "bucket_no_upscale": args.bucket_no_upscale},
      "datasets": [{"subsets": [{"image_dir": images_folder, "num_repeats": current_num_repeats, "caption_extension": args.caption_extension}]}] # Dropout не здесь
    }
    clean_dataset_dict = remove_none_recursive(dataset_dict)
    try:
        with open(dataset_config_file, "w", encoding='utf-8') as f: toml.dump(clean_dataset_dict, f)
        print(f"  Dataset config saved: {dataset_config_file}")
    except Exception as e: print(f"[!] Error writing dataset config {dataset_config_file}: {e}", file=sys.stderr); return None

    return config_file, dataset_config_file


# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 5: Generate training config files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Основные Настройки ---
    g_main = parser.add_argument_group('Main Settings')
    g_main.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    g_main.add_argument("--base-dir", type=str, default=".", help="Base directory.")
    # --- Источник Модели ---
    g_model = parser.add_argument_group('Model Source')
    g_model.add_argument("--base-model", type=str, required=True, help="REQUIRED: Base model path/URL.")
    g_model.add_argument("--custom-model", type=str, default=None, help="Override base_model.")
    g_model.add_argument("--base-vae", type=str, default=None, help="External VAE path/URL (optional).")
    g_model.add_argument("--custom-vae", type=str, default=None, help="Override base_vae.")
    g_model.add_argument("--v-pred", action='store_true', help="Use v-prediction model.")
    # --- Настройки Тренировки ---
    g_train = parser.add_argument_group('Training Settings')
    g_train.add_argument("--resolution", type=int, default=1024, help="Training resolution.")
    g_train.add_argument("--caption-extension", type=str, default=".txt", help="Caption file extension.")
    g_train.add_argument("--shuffle-tags", action=argparse.BooleanOptionalAction, default=True, help="Shuffle caption tags.")
    g_train.add_argument("--keep-tokens", type=int, default=1, help="Tokens to keep at caption start.")
    g_train.add_argument("--flip-aug", action='store_true', help="Enable flip augmentation.")
    g_train.add_argument("--num-repeats", type=int, default=None, metavar='N', help="Repeats per image (overrides --auto-repeats).")
    g_train.add_argument("--auto-repeats", action='store_true', help="Auto-determine repeats based on image count.")
    g_train.add_argument("--preferred-unit", type=str, choices=["Epochs", "Steps"], default="Epochs", help="Unit for training duration.")
    g_train.add_argument("--how-many", type=int, default=10, help="Number of epochs or steps.")
    g_train.add_argument("--save-every-n-epochs", type=int, default=1, metavar='N', help="Save checkpoint every N epochs.")
    g_train.add_argument("--keep-only-last-n-epochs", type=int, default=10, metavar='N', help="Keep only the last N saved epochs.")
    # ===> ДОБАВЛЕННЫЕ АРГУМЕНТЫ <===
    g_train.add_argument("--caption-dropout", type=float, default=0.0, metavar='RATE', help="Caption dropout rate (0-1).")
    g_train.add_argument("--caption-dropout-every-n-epochs", type=int, default=0, metavar='N', help="Apply caption dropout every N epochs (0=never).")
    # ===> КОНЕЦ ДОБАВЛЕННЫХ АРГУМЕНТОВ <===
    g_train.add_argument("--tag-dropout", type=float, default=0.0, metavar='RATE', help="Tag dropout rate (0-1).")
    # --- Параметры Обучения ---
    g_learn = parser.add_argument_group('Learning Parameters')
    g_learn.add_argument("--unet-lr", type=float, default=3e-4, help="U-Net learning rate.")
    g_learn.add_argument("--text-encoder-lr", type=float, default=6e-5, help="Text Encoder learning rate (0 to disable).")
    g_learn.add_argument("--lr-scheduler", type=str, default="cosine_with_restarts", choices=["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"], help="LR scheduler.")
    g_learn.add_argument("--lr-scheduler-num-cycles", type=int, default=3, help="Cycles for cosine_with_restarts.")
    g_learn.add_argument("--lr-scheduler-power", type=float, default=1.0, help="Power for polynomial scheduler.")
    g_learn.add_argument("--lr-warmup-ratio", type=float, default=0.05, help="LR warmup ratio (0.0-0.2).")
    g_learn.add_argument("--min-snr-gamma", type=float, default=8.0, help="Min SNR gamma (<= 0 to disable).")
    g_learn.add_argument("--noise-offset", type=float, default=0.0, help="Noise offset (0 to disable).")
    g_learn.add_argument("--ip-noise-gamma", type=float, default=0.0, help="Instance Prompt Noise Gamma (0 to disable).")
    g_learn.add_argument("--multinoise", action='store_true', help="Enable multiresolution noise.")
    g_learn.add_argument("--zero-terminal-snr", action='store_true', help="Enable zero terminal SNR.")
    # --- Структура LoRA ---
    g_lora = parser.add_argument_group('LoRA Structure')
    g_lora.add_argument("--lora-type", type=str, choices=["LoRA", "LoCon"], default="LoRA", help="LoRA network type.")
    g_lora.add_argument("--network-dim", type=int, default=32, help="Network dimension (rank).")
    g_lora.add_argument("--network-alpha", type=int, default=16, help="Network alpha.")
    g_lora.add_argument("--conv-dim", type=int, default=16, help="Conv dimension for LoCon.")
    g_lora.add_argument("--conv-alpha", type=int, default=8, help="Conv alpha for LoCon.")
    g_lora.add_argument("--continue-from-lora", type=str, default=None, metavar='PATH', help="Path to existing LoRA to continue.")
    # --- Параметры Тренировки (Технические) ---
    g_tech = parser.add_argument_group('Technical Training Parameters')
    g_tech.add_argument("--auto-vram-params", action='store_true', help="Auto-set batch_size, optimizer, precision.")
    g_tech.add_argument("--train-batch-size", type=int, default=None, metavar='N', help="Batch size (overrides auto).")
    g_tech.add_argument("--cross-attention", type=str, choices=["sdpa", "xformers"], default="sdpa", help="Cross attention implementation.")
    g_tech.add_argument("--precision", type=str, choices=["float", "fp16", "bf16", "full_fp16", "full_bf16"], default=None, metavar='TYPE', help="Training precision (overrides auto).")
    g_tech.add_argument("--cache-latents", action=argparse.BooleanOptionalAction, default=True, help="Cache latents.")
    g_tech.add_argument("--cache-latents-to-disk", action=argparse.BooleanOptionalAction, default=True, help="Cache latents to disk.")
    g_tech.add_argument("--cache-text-encoder-outputs", action='store_true', help="Cache text encoder outputs.")
    g_tech.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing.")
    g_tech.add_argument("--optimizer", type=str, default=None, choices=["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"], metavar='OPT', help="Optimizer (overrides auto).")
    g_tech.add_argument("--use-recommended-optimizer-args", action='store_true', help="Use recommended args for optimizer.")
    g_tech.add_argument("--optimizer-args", type=str, default="", help="Additional optimizer arguments.")
    g_tech.add_argument("--max-data-loader-n-workers", type=int, default=2, help="Workers for data loading.")
    g_tech.add_argument("--seed", type=int, default=42, help="Random seed.")
    g_tech.add_argument("--lowram", action='store_true', help="Enable Kohya low RAM optimizations.")
    # --- Настройки Бакетов ---
    g_bucket = parser.add_argument_group('Bucket Settings')
    g_bucket.add_argument("--bucket-reso-steps", type=int, default=64, help="Steps for bucket resolution.")
    g_bucket.add_argument("--min-bucket-reso", type=int, default=256, help="Minimum bucket resolution.")
    g_bucket.add_argument("--max-bucket-reso", type=int, default=4096, help="Maximum bucket resolution.")
    g_bucket.add_argument("--bucket-no-upscale", action='store_true', help="Disable upscaling aspect ratio buckets.")

    return parser.parse_args()


# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    paths = { "project": project_dir, "images": os.path.join(project_dir, "dataset"), "output": os.path.join(project_dir, "output"), "logs": os.path.join(project_dir, "logs"), "config": os.path.join(project_dir, "config") }

    print("--- Step 5: Generate Config Files ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Config Folder: {paths['config']}")

    # Авто-определение параметров
    gpu_vram = common_utils.get_gpu_vram()
    final_params = {}
    auto_vram_settings = {}
    if args.auto_vram_params: auto_vram_settings = common_utils.determine_vram_parameters(gpu_vram); print(f"[*] Auto VRAM params: {auto_vram_settings}")

    final_params['train_batch_size'] = args.train_batch_size if args.train_batch_size is not None else auto_vram_settings.get('batch_size', 1)
    final_params['optimizer'] = args.optimizer if args.optimizer is not None else auto_vram_settings.get('optimizer', 'AdamW8bit')
    final_params['precision'] = args.precision if args.precision is not None else auto_vram_settings.get('precision', 'fp16')
    final_params['optimizer_args'] = []
    if args.use_recommended_optimizer_args:
        optimizer_lower = final_params['optimizer'].lower()
        if optimizer_lower == "adamw8bit": final_params['optimizer_args'] = ["weight_decay=0.1", "betas=[0.9,0.99]"]
        elif optimizer_lower == "prodigy": final_params['optimizer_args'] = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
    elif args.optimizer_args: final_params['optimizer_args'] = args.optimizer_args.split()

    image_count = common_utils.get_image_count(paths['images'])
    auto_repeats_val = 0
    if args.auto_repeats: auto_repeats_val = common_utils.determine_num_repeats(image_count); print(f"[*] Auto num_repeats: {auto_repeats_val} (based on {image_count} images)")
    final_params['num_repeats'] = args.num_repeats if args.num_repeats is not None else auto_repeats_val if args.auto_repeats else 10

    print("--- Final Effective Parameters for Config ---")
    print(f"  Batch Size: {final_params['train_batch_size']}"); print(f"  Optimizer: {final_params['optimizer']}"); print(f"  Optimizer Args: {' '.join(final_params['optimizer_args']) if final_params['optimizer_args'] else 'None'}"); print(f"  Precision: {final_params['precision']}"); print(f"  Num Repeats: {final_params['num_repeats']}"); print("-------------------------------------------")

    result = generate_configuration_files(paths, args, final_params)

    if result: print("\n--- Step 5 Finished Successfully ---")
    else: print("\n--- Step 5 Failed ---"); sys.exit(1)