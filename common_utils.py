# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import re
import platform
# Импорт torch вынесен сюда, чтобы другие скрипты могли его использовать
try:
    import torch
except ImportError:
    # Не фатально здесь, но get_gpu_vram вернет 0
    print("[!] Warning: PyTorch not found in common_utils. VRAM detection might fail.", file=sys.stderr)
    torch = None

# --- Константы ---
SUPPORTED_IMG_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SUPPORTED_TAG_TYPES = (".txt", ".caption")

# --- Утилита для запуска команд ---
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
                  sys.exit(1) # Выход по check=True и ненулевому коду
        return process
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running command: {command_str}", file=sys.stderr); print(f"[!] Exit code: {e.returncode}", file=sys.stderr);
        if capture_output:
            stdout = e.stdout.decode(errors='ignore').strip() if isinstance(e.stdout, bytes) else (e.stdout.strip() if e.stdout else ""); stderr = e.stderr.decode(errors='ignore').strip() if isinstance(e.stderr, bytes) else (e.stderr.strip() if e.stderr else "")
            if stdout: print(f"[!] Stdout:\n{stdout}", file=sys.stderr);
            if stderr: print(f"[!] Stderr:\n{stderr}", file=sys.stderr);
        # check=True уже вызвал бы исключение, но на всякий случай
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

# --- Другие общие утилиты ---
def split_tags(tagstr):
    """Разделяет строку тегов по запятой."""
    if not isinstance(tagstr, str): return []
    return [s.strip() for s in tagstr.split(",") if s.strip()]

def get_gpu_vram():
    """Возвращает объем VRAM первого GPU в ГБ или 0."""
    if torch and torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(0)
            vram_gb = properties.total_memory / (1024**3)
            print(f"[*] Detected GPU: {properties.name} with {vram_gb:.2f} GB VRAM.")
            return vram_gb
        except Exception as e:
            print(f"[!] Error getting GPU info: {e}", file=sys.stderr)
            return 0
    else:
        # Не печатаем ошибку здесь, т.к. torch может быть None легально до setup
        # print("[!] No CUDA-enabled GPU found by PyTorch.")
        return 0

def get_image_count(images_folder):
    """Считает количество файлов изображений в папке."""
    if not os.path.isdir(images_folder):
        print(f"[!] Image directory not found or is not a directory: {images_folder}", file=sys.stderr)
        return 0
    try:
        count = len([f for f in os.listdir(images_folder) if f.lower().endswith(SUPPORTED_IMG_TYPES)])
        print(f"[*] Found {count} images in {images_folder}")
        return count
    except OSError as e:
        print(f"[!] Error reading image directory {images_folder}: {e}", file=sys.stderr)
        return 0

def determine_num_repeats(image_count):
    """Определяет количество повторов на основе количества изображений."""
    if image_count <= 0:
        return 10
    elif image_count <= 20:
        return 10
    elif image_count <= 30:
        return 7
    elif image_count <= 50:
        return 6
    elif image_count <= 75:
        return 5
    elif image_count <= 100:
        return 4
    elif image_count <= 200:
        return 3
    else:
        return 2

def determine_vram_parameters(vram_gb):
    """Определяет параметры на основе VRAM."""
    if vram_gb <= 0: print("[!] Cannot determine VRAM. Using conservative defaults."); return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 10: print("[*] Low VRAM (<10GB). Using memory-saving."); return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 16: print("[*] Medium VRAM (10-15GB). Using balanced."); return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 24: print("[*] Good VRAM (16-23GB). Increasing batch."); return {"batch_size": 2, "optimizer": "AdamW8bit", "precision": "fp16"}
    else: print("[*] High VRAM (24GB+). High performance."); return {"batch_size": 4, "optimizer": "AdamW8bit", "precision": "fp16"}

def get_venv_python(base_dir, venv_name):
     """Возвращает путь к python внутри venv."""
     venv_dir = os.path.join(base_dir, venv_name)
     python_exe = 'python.exe' if platform.system() == 'Windows' else 'python'
     scripts_or_bin = 'Scripts' if platform.system() == 'Windows' else 'bin'
     venv_python_path = os.path.join(venv_dir, scripts_or_bin, python_exe)
     if not os.path.exists(venv_python_path):
          print(f"[!] Venv Python not found at expected path: {venv_python_path}", file=sys.stderr)
          return None
     return venv_python_path