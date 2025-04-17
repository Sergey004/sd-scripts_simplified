# -*- coding: utf-8 -*-
import os
import argparse
import sys
# Импорт общих утилит
try:
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr); sys.exit(1)

# --- Функция запуска тренировки ---
def run_training(paths, args, config_file, dataset_config_file):
    """Запускает процесс тренировки с использованием accelerate."""
    print("\n--- Starting Training ---")
    kohya_dir = paths["kohya"]
    venv_dir = paths["venv"]
    # Получаем python из venv через утилиту
    python_executable = common_utils.get_venv_python(os.path.dirname(venv_dir), os.path.basename(venv_dir))
    if not python_executable: print("[!] Cannot run training.", file=sys.stderr); return False

    # Определяем accelerate
    accelerate_executable = os.path.join(venv_dir, 'bin', 'accelerate') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'accelerate.exe')
    accelerate_cmd_prefix = []
    if os.path.exists(accelerate_executable): accelerate_cmd_prefix = [accelerate_executable, "launch"]
    else: print("[!] Accelerate executable not found. Trying 'python -m accelerate'..."); accelerate_cmd_prefix = [python_executable, "-m", "accelerate", "launch"]

    # Определяем скрипт тренировки
    train_script = os.path.join(kohya_dir, "sdxl_train_network.py")
    if not os.path.exists(train_script):
         train_script_fallback = os.path.join(kohya_dir, "train_network.py")
         if os.path.exists(train_script_fallback): train_script = train_script_fallback; print("[*] Using 'train_network.py'.")
         else: print(f"[!] Error: Training script not found in {kohya_dir}", file=sys.stderr); return False

    # Собираем команду
    cmd = accelerate_cmd_prefix + [ "--num_cpu_threads_per_process", str(args.num_cpu_threads), train_script, f"--config_file={config_file}", f"--dataset_config={dataset_config_file}" ]

    print(f"[*] Launching training command...")
    # Используем run_cmd из common_utils
    result = common_utils.run_cmd(cmd, check=True, cwd=kohya_dir)

    if result and result.returncode == 0:
        print("\n--- Training Finished Successfully ---")
        print(f"[*] LoRA model(s) saved in: {paths['output']}")
        return True
    else:
        print("\n--- Training Failed or Exited with Errors ---")
        return False

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 6: Run LoRA training using generated config files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Используем дефисы в именах аргументов
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory.")
    parser.add_argument("--kohya-dir-name", type=str, default="kohya_ss", help="Kohya scripts directory name.")
    parser.add_argument("--venv-name", type=str, default="lora_env", help="Venv directory name.")
    parser.add_argument("--num-cpu-threads", type=int, default=2, help="CPU threads for Accelerate.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    # Доступ к аргументам через подчеркивания
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    paths = { "project": project_dir, "output": os.path.join(project_dir, "output"), "config": os.path.join(project_dir, "config"), "kohya": os.path.join(base_dir, args.kohya_dir_name), "venv": os.path.join(base_dir, args.venv_name) }

    print("--- Step 6: Run Training ---")
    print(f"[*] Project: {args.project_name}")

    # Ищем конфиг файлы
    config_file = os.path.join(paths["config"], f"training_{args.project_name}.toml")
    dataset_config_file = os.path.join(paths["config"], f"dataset_{args.project_name}.toml")

    if not os.path.exists(config_file) or not os.path.exists(dataset_config_file):
         print(f"[!] Error: Config files not found!", file=sys.stderr); print(f"  Expected: {config_file}, {dataset_config_file}"); print(f"[-] Please run Step 5 (generate_configs.py) first."); sys.exit(1)

    print(f"[*] Using Training Config: {config_file}")
    print(f"[*] Using Dataset Config: {dataset_config_file}")

    # Запускаем тренировку
    success = run_training(paths, args, config_file, dataset_config_file)

    if not success: sys.exit(1) # Выход с ошибкой