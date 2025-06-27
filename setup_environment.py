# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import sys
import re

# --- Константы (дублируем из основного скрипта для согласованности) ---
TORCH_VERSION = "2.5.1+cu124" # Убедитесь, что cu124 подходит вашей CUDA
TORCHVISION_VERSION = "0.20.1+cu124"
XFORMERS_VERSION = "0.0.29.post1"
ACCELERATE_VERSION = "0.33.0"
TRANSFORMERS_VERSION = "4.44.0"
DIFFUSERS_VERSION = "0.25.0"
KOHYA_SS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
KOHYA_SS_COMMIT = "e89653975ddf429cdf0c0fd268da0a5a3e8dba1f"

# --- Утилита для запуска команд ---
def run_cmd(command, check=True, shell=False, capture_output=False, text=False, cwd=None, env=None):
    """Утилита для запуска команд оболочки."""
    try:
        print(f"[*] Running: {' '.join(command) if isinstance(command, list) else command}")
        process = subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text, cwd=cwd, env=env)
        if capture_output:
            print(process.stdout)
            if process.stderr:
                print(process.stderr, file=sys.stderr)
        return process
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running command: {e}", file=sys.stderr)
        if capture_output:
            # Декодируем байты, если вывод не текст
            stdout = e.stdout.decode(errors='ignore') if isinstance(e.stdout, bytes) else e.stdout
            stderr = e.stderr.decode(errors='ignore') if isinstance(e.stderr, bytes) else e.stderr
            print(f"[!] Stdout: {stdout}", file=sys.stderr)
            print(f"[!] Stderr: {stderr}", file=sys.stderr)
        if check:
            sys.exit(f"Command failed with exit code {e.returncode}")
        return None
    except FileNotFoundError as e:
        print(f"[!] Error: Command not found - {command[0]}. Is it installed and in PATH?", file=sys.stderr)
        print(f"[!] Details: {e}", file=sys.stderr)
        if check:
            sys.exit("Command not found.")
        return None
    except Exception as e:
        print(f"[!] An unexpected error occurred running command: {e}", file=sys.stderr)
        if check:
            sys.exit("Unexpected error during command execution.")
        return None


# --- Функция применения патчей (копия из основного скрипта) ---
def apply_kohya_patches(kohya_dir, python_executable, load_truncated=True, better_epoch_names=True, fix_diffusers=True):
    """Применяет патчи к скриптам kohya_ss."""
    print("[*] Applying patches to kohya_ss scripts...")
    # Определяем пути внутри этой функции для ясности
    train_util_path = os.path.join(kohya_dir, 'library', 'train_util.py')
    sdxl_train_network_path = os.path.join(kohya_dir, 'sdxl_train_network.py')
    # Определяем путь к site-packages более надежно
    site_packages_path = None
    try:
        # Используем python из venv для определения site-packages
        result = run_cmd([python_executable, '-c', "import site; print(site.getsitepackages()[0])"],
                         capture_output=True, text=True, check=False)
        if result and result.returncode == 0 and result.stdout:
             site_packages_path = result.stdout.strip()
    except Exception as e:
        print(f"[!] Warning: Could not determine site-packages path automatically: {e}", file=sys.stderr)

    diffusers_deprecation_path = None
    if site_packages_path and os.path.isdir(site_packages_path):
         diffusers_deprecation_path = os.path.join(site_packages_path, 'diffusers', 'utils', 'deprecation_utils.py')
    else:
         print("[!] Warning: site-packages path not found. Cannot patch diffusers.", file=sys.stderr)


    # Патч для усеченных изображений
    if load_truncated and os.path.exists(train_util_path):
        try:
            with open(train_util_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                if 'ImageFile.LOAD_TRUNCATED_IMAGES' not in content:
                    f.seek(0)
                    f.write(content.replace('from PIL import Image', 'from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True', 1))
                    f.truncate()
                    print("  [+] Patched train_util.py for truncated images.")
                else:
                    print("  [*] Truncated images patch already applied.")
        except Exception as e:
            print(f"  [!] Failed to patch train_util.py for truncated images: {e}", file=sys.stderr)

    # Патч для имен эпох
    if better_epoch_names:
        if os.path.exists(train_util_path):
            try:
                with open(train_util_path, 'r+', encoding='utf-8') as f:
                    content = f.read()
                    new_content = content.replace('{:06d}', '{:02d}')
                    if content != new_content:
                        f.seek(0)
                        f.write(new_content)
                        f.truncate()
                        print("  [+] Patched train_util.py for shorter epoch names.")
                    else:
                         print("  [*] Shorter epoch names patch already applied to train_util.py.")
            except Exception as e:
                print(f"  [!] Failed to patch train_util.py for epoch names: {e}", file=sys.stderr)

        if os.path.exists(sdxl_train_network_path):
             try:
                 with open(sdxl_train_network_path, 'r+', encoding='utf-8') as f:
                     content = f.read()
                     pattern = r'("\." \+ args\.save_model_as\))'
                     replacement = r'"-{:02d}.".format(num_train_epochs) + args.save_model_as)'
                     new_content, count = re.subn(pattern, replacement, content)
                     if count > 0:
                         f.seek(0)
                         f.write(new_content)
                         f.truncate()
                         print(f"  [+] Patched {os.path.basename(sdxl_train_network_path)} for last epoch naming.")
                     else:
                          print(f"  [*] Last epoch naming patch already applied to {os.path.basename(sdxl_train_network_path)}.")
             except Exception as e:
                 print(f"  [!] Failed to patch {os.path.basename(sdxl_train_network_path)} for epoch naming: {e}", file=sys.stderr)

    # Патч для Diffusers deprecation warning
    if fix_diffusers and diffusers_deprecation_path and os.path.exists(diffusers_deprecation_path):
         try:
             with open(diffusers_deprecation_path, 'r+', encoding='utf-8') as f:
                 content = f.read()
                 pattern = r'(if version\.parse)'
                 replacement = r'if False:#\1'
                 new_content, count = re.subn(pattern, replacement, content)
                 if count > 0:
                     f.seek(0)
                     f.write(new_content)
                     f.truncate()
                     print("  [+] Patched diffusers deprecation_utils.py.")
                 else:
                     print("  [*] Diffusers deprecation patch already applied.")
         except Exception as e:
             print(f"  [!] Failed to patch diffusers deprecation_utils.py: {e}", file=sys.stderr)

# --- Проверка и установка aria2c ---
def ensure_aria2():
    import shutil
    if shutil.which("aria2c"):
        print("[*] aria2c is already installed.")
        return True
    print("[*] aria2c not found. Attempting to install...")
    # Определяем пакетный менеджер
    if shutil.which("apt"):
        print("[*] Installing aria2 via apt...")
        run_cmd(["sudo", "apt", "update"], check=False)
        run_cmd(["sudo", "apt", "install", "-y", "aria2"], check=False)
    elif shutil.which("dnf"):
        print("[*] Installing aria2 via dnf...")
        run_cmd(["sudo", "dnf", "install", "-y", "aria2"], check=False)
    elif shutil.which("pacman"):
        print("[*] Installing aria2 via pacman (Arch)...")
        run_cmd(["sudo", "pacman", "-Sy", "aria2", "--noconfirm"], check=False)
    elif shutil.which("emerge"):
        print("[*] Installing aria2 via emerge (Gentoo)...")
        run_cmd(["sudo", "emerge", "-av", "aria2"], check=False)
    else:
        print("[!] No supported package manager found (apt, dnf, pacman, emerge). Please install aria2 manually.", file=sys.stderr)
        return False
    # Проверяем снова
    if shutil.which("aria2c"):
        print("[+] aria2c installed successfully.")
        return True
    print("[!] Failed to install aria2c automatically. Please install it manually.", file=sys.stderr)
    return False

# --- Основная функция установки ---
def setup_venv_and_install(base_dir, venv_name, kohya_dir_name):
    """Создает venv (если нет) и устанавливает все зависимости."""
    base_dir = os.path.abspath(base_dir)
    venv_dir = os.path.join(base_dir, venv_name)
    kohya_dir = os.path.join(base_dir, kohya_dir_name)
    kohya_req_file = os.path.join(kohya_dir, 'requirements.txt')
    temp_req_file = os.path.join(kohya_dir, 'requirements_temp.txt')

    print(f"[*] Base directory: {base_dir}")
    print(f"[*] Virtual env directory: {venv_dir}")
    print(f"[*] Kohya scripts directory: {kohya_dir}")

    # 1. Создание venv, если его нет
    activate_script_unix = os.path.join(venv_dir, 'bin', 'activate')
    activate_script_win = os.path.join(venv_dir, 'Scripts', 'activate')
    if not os.path.exists(activate_script_unix) and not os.path.exists(activate_script_win):
        print(f"[*] Virtual environment not found. Creating in {venv_dir}...")
        # Используем системный python для создания venv
        run_cmd([sys.executable, '-m', 'venv', venv_dir], check=True)
        print("[+] Virtual environment created.")
    else:
        print("[*] Virtual environment already exists.")

    # 2. Определение путей к pip и python внутри venv
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'pip.exe')
    python_executable = os.path.join(venv_dir, 'bin', 'python') if sys.platform != 'win32' else os.path.join(venv_dir, 'Scripts', 'python.exe')

    if not os.path.exists(pip_executable):
        print(f"[!] Pip executable not found in venv: {pip_executable}. Venv creation might have failed.", file=sys.stderr)
        sys.exit(1)

    # 3. Установка основных зависимостей (Torch, xFormers, etc.)
    print("\n[*] Installing core dependencies into venv...")
    run_cmd([pip_executable, "install", "-U", "pip", "setuptools", "wheel"], check=True) # Обновляем pip/setuptools
    run_cmd([pip_executable, "install",
             f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"xformers=={XFORMERS_VERSION}",
             "--index-url", "https://download.pytorch.org/whl/cu124"], # ЗАМЕНИТЕ cu124 если нужно
             check=True)
    run_cmd([pip_executable, "install", "onnx", "onnxruntime-gpu", "--extra-index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"], check=True)
    run_cmd([pip_executable, "install",
             f"accelerate=={ACCELERATE_VERSION}", f"transformers=={TRANSFORMERS_VERSION}", f"diffusers[torch]=={DIFFUSERS_VERSION}",
             "bitsandbytes==0.44.0", "safetensors==0.4.4", "prodigyopt==1.0", "lion-pytorch==0.0.6", "schedulefree==1.4",
             "toml==0.10.2", "einops==0.7.0", "ftfy==6.1.1", "opencv-python==4.8.1.78", "pytorch-lightning==1.9.0",
             "wandb", "scipy", "requests", # requests нужен для скачивания в основном скрипте
             "fiftyone", "scikit-learn", "timm", "fairscale" # Добавим fiftyone и sklearn для дедупликации
            ], check=True)
    run_cmd([pip_executable, "install" ,"https://huggingface.co/spaces/cocktailpeanut/gradio_logsview/resolve/main/gradio_logsview-0.0.17-py3-none-any.whl"])
    print("[+] Core dependencies installed.")

    # 4. Клонирование/Обновление Kohya-ss
    print(f"\n[*] Checking Kohya sd-scripts in {kohya_dir}...")
    if not os.path.exists(os.path.join(kohya_dir, '.git')):
        print(f"[*] Cloning kohya_ss sd-scripts...")
        run_cmd(['git', 'clone', KOHYA_SS_REPO, kohya_dir], check=True)
        print(f"[*] Checking out specific commit: {KOHYA_SS_COMMIT}")
        run_cmd(['git', 'checkout', KOHYA_SS_COMMIT], check=True, cwd=kohya_dir)
    else:
        print("[*] Kohya directory exists. Verifying commit...")
        try:
            current_commit = run_cmd(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, cwd=kohya_dir).stdout.strip()
            if current_commit != KOHYA_SS_COMMIT:
                print(f"[*] Current commit is {current_commit}. Checking out {KOHYA_SS_COMMIT}...")
                run_cmd(['git', 'fetch'], check=True, cwd=kohya_dir) # Fetch first
                run_cmd(['git', 'checkout', KOHYA_SS_COMMIT], check=True, cwd=kohya_dir)
            else:
                print(f"[*] Correct commit ({KOHYA_SS_COMMIT}) already checked out.")
        except Exception as e:
            print(f"[!] Error verifying/checking out commit: {e}. Please check git status in {kohya_dir} manually.", file=sys.stderr)

    # 5. Установка зависимостей Kohya-ss
    print("\n[*] Installing Kohya sd-scripts requirements...")
    if os.path.exists(kohya_req_file):
        # Фильтруем уже установленные основные пакеты
        lines_to_install = []
        core_packages = {'torch', 'torchvision', 'xformers', 'accelerate', 'transformers', 'diffusers', 'bitsandbytes', 'safetensors'}
        try:
            with open(kohya_req_file, 'r', encoding='utf-8') as infile, open(temp_req_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    line_strip = line.strip()
                    # Пропускаем пустые, комментарии, editable install и основные пакеты
                    if not line_strip or line_strip.startswith('#') or line_strip.startswith('-e'):
                        continue
                    # Получаем имя пакета (до ==, >=, <, [)
                    package_name = re.split(r'[=<>\[]', line_strip)[0].strip()
                    if package_name and package_name not in core_packages:
                        outfile.write(line)
                        lines_to_install.append(package_name)

            if lines_to_install:
                print(f"[*] Installing from filtered requirements: {', '.join(lines_to_install)}")
                run_cmd([pip_executable, 'install', '-r', temp_req_file], check=True)
            else:
                print("[*] No additional requirements found in filtered file.")

            if os.path.exists(temp_req_file):
                os.remove(temp_req_file)

            # Установка специфичных вещей kohya, если они есть и не в requirements (пример)
            setup_py_path = os.path.join(kohya_dir, 'setup.py')
            if os.path.exists(setup_py_path):
                run_cmd([pip_executable, 'install', '-e', '.'], check=True, cwd=kohya_dir) # Установка kohya_ss из локального каталога
            print("[*] Kohya ss scripts installed.")
        except FileNotFoundError as e:
            print(f"[!] Error: {e}. File not found. Please check the path.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running command: {e}. Command might have failed.", file=sys.stderr)
        except PermissionError as e:
            print(f"[!] Permission error: {e}. Please check your permissions.", file=sys.stderr)
        except OSError as e:
            print(f"[!] OS error: {e}. Please check your system configuration.", file=sys.stderr)
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}", file=sys.stderr)
            print("[+] Kohya_ss requirements installed.")
        except Exception as e:
            print(f"[!] Error processing or installing kohya requirements: {e}", file=sys.stderr)
            if os.path.exists(temp_req_file):
                os.remove(temp_req_file) # Удаляем временный файл при ошибке
    else:
        print(f"[!] Warning: {kohya_req_file} not found. Cannot install kohya_ss requirements.", file=sys.stderr)
    
    ensure_aria2()
    # 6. Применение патчей
    apply_kohya_patches(kohya_dir, python_executable)

    print("\n--- Environment Setup Complete ---")
    print(f"Virtual environment '{venv_name}' is ready in '{base_dir}'.")
    print("Please activate it before running the main training script:")
    print(f"  Linux/macOS: source {activate_script_unix}")
    print(f"  Windows CMD: .\\{activate_script_win}")
    print(f"  Windows PowerShell: .\\{activate_script_win}.ps1")


# --- Парсер аргументов для setup ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Setup Python Virtual Environment and Install Dependencies for LoRA Training")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory to create venv and clone kohya_ss.")
    parser.add_argument("--venv_name", type=str, default="lora_env", help="Name for the virtual environment directory.")
    parser.add_argument("--kohya_dir_name", type=str, default="kohya_ss", help="Name for the kohya_ss scripts directory.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    setup_venv_and_install(args.base_dir, args.venv_name, args.kohya_dir_name)