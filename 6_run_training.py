# -*- coding: utf-8 -*-
import os
import argparse
import subprocess
import sys

# --- Утилита run_cmd (копия) ---
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

# --- Функция запуска тренировки ---
def run_training(paths, args, config_file, dataset_config_file):
    """Запускает процесс тренировки с использованием accelerate."""
    print("\n--- Starting Training ---")
    kohya_dir = paths["kohya"]
    venv_dir = paths["venv"]
    python_executable = os.path.join(venv_dir, 'bin', 'python') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'python.exe')
    accelerate_executable = os.path.join(venv_dir, 'bin', 'accelerate') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'accelerate.exe')

    # Проверяем accelerate
    accelerate_cmd_prefix = []
    if os.path.exists(accelerate_executable):
        accelerate_cmd_prefix = [accelerate_executable, "launch"]
    elif os.path.exists(python_executable):
         print("[!] Accelerate executable not found. Trying 'python -m accelerate'...")
         accelerate_cmd_prefix = [python_executable, "-m", "accelerate", "launch"]
    else:
         print(f"[X] CRITICAL ERROR: Neither accelerate nor python found in venv: {venv_dir}", file=sys.stderr)
         return False

    # Определяем скрипт тренировки
    train_script = os.path.join(kohya_dir, "sdxl_train_network.py")
    if not os.path.exists(train_script):
         train_script_fallback = os.path.join(kohya_dir, "train_network.py")
         if os.path.exists(train_script_fallback): train_script = train_script_fallback; print("[*] Using 'train_network.py'.")
         else: print(f"[!] Error: Training script not found in {kohya_dir}", file=sys.stderr); return False

    # Собираем команду
    cmd = accelerate_cmd_prefix + [
        "--num_cpu_threads_per_process", str(args.num_cpu_threads),
        train_script,
        f"--config_file={config_file}",
        f"--dataset_config={dataset_config_file}"
    ]

    print(f"[*] Launching training command...")
    # Запускаем из папки kohya_ss
    result = run_cmd(cmd, check=True, cwd=kohya_dir) # check=True - тренировка должна завершиться успешно

    if result and result.returncode == 0:
        print("\n--- Training Finished Successfully ---")
        print(f"[*] LoRA model(s) saved in: {paths['output']}")
        return True
    else:
        print("\n--- Training Failed or Exited with Errors ---")
        return False


# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 6: Run LoRA training using generated config files.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing project, kohya_ss, venv.")
    parser.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name of the kohya_ss directory.")
    parser.add_argument("--venv_name", type=str, default="lora_env", help="Name of the virtual environment directory.")
    parser.add_argument("--num_cpu_threads", type=int, default=2, help="CPU threads per process for Accelerate.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    paths = {
        "project": project_dir,
        "output": os.path.join(project_dir, "output"), # Нужен для финального сообщения
        "config": os.path.join(project_dir, "config"),
        "kohya": os.path.join(base_dir, args.kohya_dir_name),
        "venv": os.path.join(base_dir, args.venv_name)
    }

    print("--- Step 6: Run Training ---")
    print(f"[*] Project: {args.project_name}")

    # Ищем конфиг файлы
    config_file = os.path.join(paths["config"], f"training_{args.project_name}.toml")
    dataset_config_file = os.path.join(paths["config"], f"dataset_{args.project_name}.toml")

    if not os.path.exists(config_file) or not os.path.exists(dataset_config_file):
         print(f"[!] Error: Config files not found!", file=sys.stderr)
         print(f"  Expected Training Config: {config_file}")
         print(f"  Expected Dataset Config: {dataset_config_file}")
         print(f"[-] Please run Step 5 (generate_configs.py) first.")
         sys.exit(1)

    print(f"[*] Using Training Config: {config_file}")
    print(f"[*] Using Dataset Config: {dataset_config_file}")

    # Запускаем тренировку
    success = run_training(paths, args, config_file, dataset_config_file)

    if not success:
        sys.exit(1) # Выход с ошибкой, если тренировка не удалась