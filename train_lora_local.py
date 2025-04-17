# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import toml
import re
import json
import time
import sys
from urllib.request import urlopen, Request
from pathlib import Path
from collections import Counter
import math

# Попытка импорта torch для определения VRAM
try:
    import torch
except ImportError:
    print("[!] Warning: PyTorch not found. VRAM auto-detection will not work until dependencies are installed.", file=sys.stderr)
    torch = None

# --- Константы ---
TORCH_VERSION = "2.5.1+cu124" # Убедитесь, что cu124 подходит вашей CUDA
TORCHVISION_VERSION = "0.20.1+cu124"
XFORMERS_VERSION = "0.0.29.post1"
ACCELERATE_VERSION = "0.33.0"
TRANSFORMERS_VERSION = "4.44.0"
DIFFUSERS_VERSION = "0.25.0"
KOHYA_SS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
KOHYA_SS_COMMIT = "e89653975ddf429cdf0c0fd268da0a5a3e8dba1f"
SUPPORTED_IMG_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SUPPORTED_TAG_TYPES = (".txt", ".caption")

# Отключаем некоторые предупреждения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# --- Утилиты ---
def run_cmd(command, check=True, shell=False, capture_output=False, text=False, cwd=None):
    """Утилита для запуска команд оболочки."""
    try:
        print(f"[*] Running: {' '.join(command) if isinstance(command, list) else command}")
        result = subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text, cwd=cwd)
        if capture_output:
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running command: {e}", file=sys.stderr)
        if capture_output:
            print(f"[!] Stdout: {e.stdout}", file=sys.stderr)
            print(f"[!] Stderr: {e.stderr}", file=sys.stderr)
        if check:
            sys.exit(1)
        return None
    except FileNotFoundError as e:
        print(f"[!] Error: Command not found - {command[0]}. Is it installed and in PATH?", file=sys.stderr)
        print(f"[!] Details: {e}", file=sys.stderr)
        if check:
            sys.exit(1)
        return None

def split_tags(tagstr):
    """Разделяет строку тегов по запятой."""
    return [s.strip() for s in tagstr.split(",") if s.strip()]

# --- Новые Утилиты ---
def get_gpu_vram():
    """Возвращает объем VRAM первого GPU в ГБ или 0, если CUDA недоступна."""
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
        print("[!] No CUDA-enabled GPU found by PyTorch.")
        return 0

def get_image_count(images_folder):
    """Считает количество файлов изображений в папке."""
    if not os.path.isdir(images_folder):
        print(f"[!] Image directory not found: {images_folder}", file=sys.stderr)
        return 0
    # Используем глобальную константу
    count = len([f for f in os.listdir(images_folder) if f.lower().endswith(SUPPORTED_IMG_TYPES)])
    print(f"[*] Found {count} images in {images_folder}")
    return count

def determine_num_repeats(image_count):
    """Определяет количество повторов на основе количества изображений."""
    if image_count <= 0: return 10
    if image_count <= 20: return 10
    elif image_count <= 30: return 7
    elif image_count <= 50: return 6
    elif image_count <= 75: return 5
    elif image_count <= 100: return 4
    elif image_count <= 200: return 3
    else: return 2

def determine_vram_parameters(vram_gb):
    """Определяет параметры на основе VRAM."""
    if vram_gb <= 0:
        print("[!] Cannot determine VRAM parameters. Using conservative defaults.")
        return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 10:
        print("[*] Low VRAM detected (<10GB). Using memory-saving settings.")
        return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 16:
        print("[*] Medium VRAM detected (10-15GB). Using balanced settings.")
        return {"batch_size": 1, "optimizer": "AdamW8bit", "precision": "fp16"}
    elif vram_gb < 24:
        print("[*] Good VRAM detected (16-23GB). Increasing batch size.")
        return {"batch_size": 2, "optimizer": "AdamW8bit", "precision": "fp16"}
    else: # 24+ GB
        print("[*] High VRAM detected (24GB+). Using high performance settings.")
        return {"batch_size": 4, "optimizer": "AdamW8bit", "precision": "fp16"}

# --- Функции проверки зависимостей и патчей ---
def check_and_install_dependencies(kohya_dir, venv_dir):
    """Проверяет и устанавливает kohya_ss и зависимости."""
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.path.exists(os.path.join(venv_dir, 'bin')) else os.path.join(venv_dir, 'Scripts', 'pip.exe')
    python_executable = os.path.join(venv_dir, 'bin', 'python') if os.path.exists(os.path.join(venv_dir, 'bin')) else os.path.join(venv_dir, 'Scripts', 'python.exe')

    if not os.path.exists(pip_executable):
        print(f"[!] Pip not found in venv: {pip_executable}. Please ensure the virtual environment is set up correctly.", file=sys.stderr)
        sys.exit(1)

    print("[*] Checking/Installing main dependencies...")
    try:
        # Просто проверяем импорт основных библиотек
        import torch
        import torchvision
        import xformers
        import accelerate
        import transformers
        import diffusers
        print("[+] PyTorch, xFormers, Accelerate, Transformers, Diffusers seem installed.")
    except ImportError:
        print("[!] Core libraries not found. Installing...")
        # Установка Torch/Torchvision/Xformers
        run_cmd([pip_executable, "install",
                 f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}", f"xformers=={XFORMERS_VERSION}",
                 "--index-url", "https://download.pytorch.org/whl/cu124"], check=True) # ЗАМЕНИТЕ cu124 если нужно
        # Установка остальных
        run_cmd([pip_executable, "install",
                 f"accelerate=={ACCELERATE_VERSION}", f"transformers=={TRANSFORMERS_VERSION}", f"diffusers[torch]=={DIFFUSERS_VERSION}",
                 "bitsandbytes==0.44.0", "safetensors==0.4.4", "prodigyopt==1.0", "lion-pytorch==0.0.6", "schedulefree==1.4",
                 "toml==0.10.2", "einops==0.7.0", "ftfy==6.1.1", "opencv-python==4.8.1.78", "pytorch-lightning==1.9.0",
                 "wandb", "scipy"
                 ], check=True)

    if not os.path.exists(os.path.join(kohya_dir, '.git')):
        print(f"[*] Cloning kohya_ss sd-scripts into {kohya_dir}...")
        run_cmd(['git', 'clone', KOHYA_SS_REPO, kohya_dir], check=True)
        print(f"[*] Checking out specific commit: {KOHYA_SS_COMMIT}")
        run_cmd(['git', 'checkout', KOHYA_SS_COMMIT], check=True, cwd=kohya_dir)
        print("[*] Installing kohya_ss requirements...")
        req_file = os.path.join(kohya_dir, 'requirements.txt')
        temp_req_file = os.path.join(kohya_dir, 'requirements_temp.txt')
        if os.path.exists(req_file):
            # Фильтруем уже установленные основные пакеты
            with open(req_file, 'r') as infile, open(temp_req_file, 'w') as outfile:
                for line in infile:
                    line_strip = line.strip()
                    if not line_strip.startswith(('torch', 'torchvision', 'xformers', '-e .', '#')): # Игнорируем комментарии тоже
                         if line_strip: # Пропускаем пустые строки
                             outfile.write(line)
            run_cmd([pip_executable, 'install', '-r', temp_req_file], check=True)
            os.remove(temp_req_file)
            # Дополнительные установки (если нужны, как в Colab)
            # run_cmd([pip_executable, 'install', '-e', '../custom_scheduler'], check=False, cwd=kohya_dir) # Пример
            print("[+] Kohya_ss requirements installed.")
        else:
            print(f"[!] Warning: {req_file} not found. Cannot install kohya_ss requirements automatically.", file=sys.stderr)
    else:
        print("[+] Kohya_ss directory found.")
        # Можно добавить проверку коммита и checkout, если нужно

    # Применяем патчи
    apply_kohya_patches(kohya_dir, python_executable)


def apply_kohya_patches(kohya_dir, python_executable, load_truncated=True, better_epoch_names=True, fix_diffusers=True):
    """Применяет патчи к скриптам kohya_ss."""
    print("[*] Applying patches to kohya_ss scripts...")
    train_util_path = os.path.join(kohya_dir, 'library', 'train_util.py')
    sdxl_train_network_path = os.path.join(kohya_dir, 'sdxl_train_network.py')
    diffusers_deprecation_path = os.path.join(os.path.dirname(python_executable), '..', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'diffusers', 'utils', 'deprecation_utils.py')

    # Патч для усеченных изображений
    if load_truncated and os.path.exists(train_util_path):
        try: # Обернем в try-except на случай проблем с чтением/правами
            if 'ImageFile.LOAD_TRUNCATED_IMAGES' not in open(train_util_path).read():
                # Используем Python для замены, чтобы избежать проблем с sed на разных ОС
                with open(train_util_path, 'r+') as f:
                    content = f.read()
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
                with open(train_util_path, 'r+') as f:
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
                with open(sdxl_train_network_path, 'r+') as f:
                    content = f.read()
                    # Используем re.sub для более точной замены
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
    if fix_diffusers and os.path.exists(diffusers_deprecation_path):
         try:
             with open(diffusers_deprecation_path, 'r+') as f:
                 content = f.read()
                 pattern = r'(if version\.parse)'
                 replacement = r'if False:#\1' # Комментируем строку, сохраняя оригинал
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

# --- Функции Этапов ---
def setup_environment(base_dir, project_name, venv_name="lora_env"):
    """Создает папки проекта и настраивает виртуальное окружение."""
    project_dir = os.path.join(base_dir, project_name)
    images_folder = os.path.join(project_dir, "dataset")
    output_folder = os.path.join(project_dir, "output")
    log_folder = os.path.join(project_dir, "logs")
    config_folder = os.path.join(project_dir, "config")
    kohya_scripts_dir = os.path.join(base_dir, "kohya_ss")
    venv_dir = os.path.join(base_dir, venv_name)

    for dir_path in [project_dir, images_folder, output_folder, log_folder, config_folder, kohya_scripts_dir, venv_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Настройка Venv (если не активен)
    if sys.prefix == sys.base_prefix:
        activate_script_unix = os.path.join(venv_dir, 'bin', 'activate')
        activate_script_win = os.path.join(venv_dir, 'Scripts', 'activate')
        if not os.path.exists(activate_script_unix) and not os.path.exists(activate_script_win):
             # Создаем venv только если его нет
             print(f"[*] Creating virtual environment in {venv_dir}...")
             run_cmd([sys.executable, '-m', 'venv', venv_dir], check=True)
             print(f"[+] Virtual environment created.")

        # Сообщение пользователю о необходимости активации
        print(f"[!] Virtual environment not active. Please activate it:")
        print(f"  Linux/macOS: source {activate_script_unix}")
        print(f"  Windows CMD: .\\{activate_script_win}")
        print(f"  Windows PowerShell: .\\{activate_script_win}.ps1")
        print(f"[*] Then re-run the script to install dependencies.")
        sys.exit(0) # Выход, чтобы пользователь активировал и перезапустил
    else:
         print("[+] Virtual environment is active.")
         # Теперь можно безопасно проверять и ставить зависимости
         # Эта функция должна быть определена ДО вызова setup_environment, что исправлено
         check_and_install_dependencies(kohya_scripts_dir, venv_dir)

    return {
        "project": project_dir,
        "images": images_folder,
        "output": output_folder,
        "logs": log_folder,
        "config": config_folder,
        "kohya": kohya_scripts_dir,
        "venv": venv_dir
    }


def scrape_images(tags, images_folder, config_folder, project_name, max_resolution=3072, include_parents=True, limit=1000):
    """Скачивает изображения с Gelbooru по тегам."""
    print("\n--- Image Scraping (Gelbooru) ---")
    if not tags:
        print("[!] No tags provided for scraping. Skipping.")
        return

    try:
        import requests
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "requests"], check=True)
        import requests

    aria2c_executable = "aria2c"
    use_aria = False
    try:
        # Проверяем доступность aria2c без вывода версии
        subprocess.run([aria2c_executable, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        use_aria = True
        print("[*] aria2c found, using for faster downloads.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[!] aria2c not found or not working. Downloads will be slower via requests. Install 'aria2' for speed.")


    tags_str = tags.replace(" ", "+").replace(":", "%3a").replace("&", "%26").replace("(", "%28").replace(")", "%29")
    api_url_template = "https://gelbooru.com/index.php?page=dapi&json=1&s=post&q=index&limit=100&tags={}&pid={}"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    all_image_urls = set()
    pid = 0
    fetched_count = 0

    print(f"[*] Fetching image list for tags: {tags} (limit: {limit})")
    while fetched_count < limit:
        url = api_url_template.format(tags_str, pid)
        print(f"  Fetching page {pid+1}...")
        try:
            response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data or "post" not in data or not data["post"]:
                print(f"  No more images found on page {pid+1}.")
                break

            count_on_page = 0
            for post in data["post"]:
                 if not all(k in post for k in ['width', 'height', 'file_url', 'sample_url', 'parent_id']):
                     print(f"  [!] Skipping post with missing keys: {post.get('id', 'N/A')}")
                     continue

                 if post['parent_id'] == 0 or include_parents:
                     img_url = post['file_url'] if post['width'] * post['height'] <= max_resolution**2 else post['sample_url']
                     # Используем глобальную константу
                     if img_url.lower().endswith(SUPPORTED_IMG_TYPES):
                        all_image_urls.add(img_url)
                        count_on_page += 1
                        fetched_count += 1
                        if fetched_count >= limit: break

            print(f"  Found {count_on_page} valid images on page {pid+1}. Total fetched: {fetched_count}")
            if fetched_count >= limit: break

            pid += 1
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"[!] Error fetching page {pid+1}: {e}", file=sys.stderr)
            time.sleep(2)
            continue
        except json.JSONDecodeError as e:
            print(f"[!] Error decoding JSON from page {pid+1}: {e}", file=sys.stderr)
            print(f"  Response text: {response.text[:200]}...")
            break

    if not all_image_urls:
        print("[!] No images found matching the criteria.")
        return

    print(f"[*] Found {len(all_image_urls)} unique image URLs.")
    scrape_file = os.path.join(config_folder, f"scrape_{project_name}.txt")
    with open(scrape_file, "w") as f:
        f.write("\n".join(all_image_urls))
    print(f"[*] Image URLs saved to {scrape_file}")

    print(f"[*] Downloading images to {images_folder}...")
    if use_aria:
        run_cmd([aria2c_executable,
                 "--console-log-level=warn", "-c", "-x", "8", "-s", "8", "-k", "1M",
                 "-d", images_folder, "-i", scrape_file], check=False) # check=False чтобы не падать если aria что-то не скачает
    else:
        for i, img_url in enumerate(all_image_urls):
            try:
                filename = os.path.basename(img_url.split('?')[0])
                filepath = os.path.join(images_folder, filename)
                if os.path.exists(filepath):
                     # print(f"  Skipping {i+1}/{len(all_image_urls)}: {filename} (exists)")
                     continue # Не печатаем пропуск для чистоты лога

                print(f"  Downloading {i+1}/{len(all_image_urls)}: {filename}...")
                img_response = requests.get(img_url, headers={"User-Agent": user_agent}, stream=True, timeout=60)
                img_response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192*10): # Увеличим чанк
                        f.write(chunk)
                time.sleep(0.1)
            except requests.exceptions.RequestException as e:
                print(f"  [!] Error downloading {filename}: {e}", file=sys.stderr)
            except Exception as e:
                 print(f"  [!] Unexpected error downloading {filename}: {e}", file=sys.stderr)

    downloaded_count = get_image_count(images_folder) # Используем функцию для подсчета
    print(f"[+] Download attempt complete. Total images in dataset folder: {downloaded_count}")


def detect_duplicates(images_folder, threshold=0.985):
    """Обнаруживает дубликаты с помощью FiftyOne (если установлен)."""
    print("\n--- Duplicate Detection ---")
    try:
        import fiftyone as fo
        from fiftyone import ViewField as F
        import numpy as np
        # sklearn нужен для cosine_similarity
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("[!] scikit-learn not found. Installing...")
            run_cmd([sys.executable, "-m", "pip", "install", "scikit-learn"], check=True)
            from sklearn.metrics.pairwise import cosine_similarity
        print("[*] FiftyOne and scikit-learn found. Detecting duplicates...")
    except ImportError:
        print("[!] FiftyOne not found (`pip install fiftyone fiftyone-db-ubuntu2204`). Skipping duplicate detection.")
        return

    # Используем глобальную константу
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(SUPPORTED_IMG_TYPES)]
    if len(image_files) < 2:
        print("[!] Less than 2 images found. Skipping duplicate detection.")
        return

    dataset = None # Инициализируем переменную
    try:
        dataset_name = f"dedupe_check_{os.path.basename(images_folder)}_{int(time.time())}"
        if fo.dataset_exists(dataset_name):
            fo.delete_dataset(dataset_name) # Удаляем старый, если остался

        dataset = fo.Dataset.from_images(image_files, name=dataset_name, persistent=False)

        print("[*] Computing embeddings (may download CLIP model)...")
        model = fo.zoo.load_zoo_model("clip-vit-base32-torch") # Модель из Colab
        embeddings = dataset.compute_embeddings(model, batch_size=16)

        print("[*] Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)

        print(f"[*] Finding duplicates with threshold > {threshold}...")
        id_map = [s.id for s in dataset.select_fields(["id"])]
        samples_to_tag_delete = set()
        samples_to_tag_duplicate = set() # Keep track of originals that have duplicates

        for idx, sample in enumerate(dataset):
            if sample.id not in samples_to_tag_delete:
                # Find indices of duplicates above threshold
                dup_indices = np.where(similarity_matrix[idx] > threshold)[0]
                if len(dup_indices) > 0:
                    samples_to_tag_duplicate.add(sample.id) # Mark this one as having duplicates
                    # Mark all actual duplicates for deletion
                    for dup_idx in dup_indices:
                        dup_id = id_map[dup_idx]
                        if dup_id != sample.id: # Don't mark self for deletion
                            samples_to_tag_delete.add(dup_id)

        # Apply tags in a separate loop for clarity
        delete_count = 0
        duplicate_count = 0
        for sample in dataset:
            if sample.id in samples_to_tag_delete:
                sample.tags.append("delete")
                sample.save()
                delete_count += 1
            elif sample.id in samples_to_tag_duplicate: # Mark the original only if it's not marked for deletion itself
                sample.tags.append("has_duplicates")
                sample.save()
                duplicate_count += 1

        print(f"[+] Marked {delete_count} images with 'delete' tag.")
        print(f"[+] Marked {duplicate_count} images (kept) with 'has_duplicates' tag.")
        print("[!] IMPORTANT: This script only TAGS duplicates using FiftyOne metadata.")
        print("  You need to MANUALLY delete the files corresponding to images tagged 'delete'.")

    except Exception as e:
        print(f"[!] An error occurred during duplicate detection: {e}", file=sys.stderr)
    finally:
         # Удаление временного датасета, если он был создан и существует
         if dataset and fo.dataset_exists(dataset.name):
             try:
                 fo.delete_dataset(dataset.name)
                 print(f"[*] Deleted temporary FiftyOne dataset '{dataset.name}'.")
             except Exception as e_del:
                 print(f"[!] Error deleting temporary dataset '{dataset.name}': {e_del}", file=sys.stderr)


def tag_images(images_folder, kohya_dir, method="wd14", batch_size=8, threshold=0.35,
               caption_ext=".txt", blacklist_tags="", min_len=10, max_len=75, overwrite=False):
    """Тегирует изображения с помощью WD14 Tagger или BLIP."""
    print("\n--- Image Tagging ---")
    kohya_venv_python = os.path.join(kohya_dir, "venv", "bin", "python") if os.path.exists(os.path.join(kohya_dir, "venv", "bin")) else os.path.join(kohya_dir, "venv", "Scripts", "python.exe")

    existing_tags = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
    if existing_tags and not overwrite:
        print(f"[*] Found existing tag files ({caption_ext}). Skipping tagging. Use --overwrite_tags to re-tag.")
        return

    if overwrite and existing_tags:
         print(f"[*] Overwriting {len(existing_tags)} existing tag files...")
         for tag_file in existing_tags:
              try:
                  os.remove(os.path.join(images_folder, tag_file))
              except OSError as e:
                  print(f"  [!] Error removing {tag_file}: {e}", file=sys.stderr)


    if method == "wd14":
        script_path = os.path.join(kohya_dir, "finetune", "tag_images_by_wd14_tagger.py")
        model_repo = "SmilingWolf/wd-v1-4-swinv2-tagger-v2" # WD14 Tagger V2
        if not os.path.exists(script_path):
            print(f"[!] WD14 Tagger script not found at {script_path}. Ensure kohya_ss is correctly cloned. Skipping.", file=sys.stderr)
            return

        print(f"[*] Running WD14 Tagger (model: {model_repo}, threshold: {threshold})...")
        cmd = [
            kohya_venv_python, script_path,
            images_folder,
            "--repo_id", model_repo,
            "--model_dir", os.path.join(kohya_dir, "wd14_models_cache"), # Папка для кеша модели
            "--thresh", str(threshold),
            "--batch_size", str(batch_size),
            "--caption_extension", caption_ext,
            "--force_download", # Скачать модель если нет
            # "--remove_underscore" # Kohya скрипт может делать это сам
        ]
        run_cmd(cmd, check=True)

        print("[*] Post-processing WD14 tags (applying blacklist)...")
        blacklisted_tags = set(split_tags(blacklist_tags))
        top_tags = Counter()
        tag_files = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]

        for txt_file in tag_files:
            filepath = os.path.join(images_folder, txt_file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f: # Укажем кодировку
                    current_tags = split_tags(f.read())

                # Kohya скрипт должен был уже убрать подчеркивания, применяем только блеклист
                processed_tags = [tag for tag in current_tags if tag not in blacklisted_tags]
                top_tags.update(processed_tags)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(", ".join(processed_tags))
            except Exception as e:
                 print(f"  [!] Error post-processing tag file {txt_file}: {e}", file=sys.stderr)

        print("[+] WD14 Tagging and post-processing complete.")
        if top_tags:
            print("  Top 50 tags found (after blacklist):")
            for tag, count in top_tags.most_common(50):
                print(f"    {tag} ({count})")

    elif method == "blip":
        script_path = os.path.join(kohya_dir, "finetune", "make_captions.py")
        if not os.path.exists(script_path):
            print(f"[!] BLIP captioning script not found at {script_path}. Skipping.", file=sys.stderr)
            return

        print(f"[*] Running BLIP Captioning (min: {min_len}, max: {max_len})...")
        cmd = [
            kohya_venv_python, script_path,
            images_folder,
            "--batch_size", "1",
            "--min_length", str(min_len),
            "--max_length", str(max_len),
            "--caption_extension", caption_ext,
            "--max_data_loader_n_workers", "2"
        ]
        run_cmd(cmd, check=True)
        print("[+] BLIP Captioning complete.")
    else:
        print(f"[!] Unknown tagging method: {method}. Skipping.")


def curate_tags(images_folder, activation_tag, remove_tags, search_tags, replace_tags, caption_ext=".txt", sort_alpha=False, remove_duplicates=False):
    """Обрабатывает файлы тегов: добавляет активационный тег, удаляет ненужные, ищет/заменяет."""
    print("\n--- Tag Curation ---")
    tag_files = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
    if not tag_files:
        print("[!] No tag files found to curate.")
        return

    activation_tag_list = split_tags(activation_tag)
    remove_tags_list = set(split_tags(remove_tags))
    search_tags_list = split_tags(search_tags) # Оставляем как список для порядка? Нет, set для поиска лучше.
    search_tags_set = set(search_tags_list)
    replace_with_list = split_tags(replace_tags)

    remove_count = 0
    replace_file_count = 0 # Считаем файлы, где была замена
    activation_added_count = 0

    print(f"[*] Curating tags in {len(tag_files)} files...")
    for txt_file in tag_files:
        filepath = os.path.join(images_folder, txt_file)
        processed = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tags = split_tags(f.read()) # Работаем со списком

            original_tag_count = len(tags)
            current_tag_set = set(tags) # Множество для быстрого поиска

            # 1. Удаление тегов из remove_list
            tags_after_remove = [t for t in tags if t not in remove_tags_list]
            removed_count_this_file = original_tag_count - len(tags_after_remove)
            if removed_count_this_file > 0:
                remove_count += removed_count_this_file
                tags = tags_after_remove
                current_tag_set = set(tags) # Обновляем set
                processed = True

            # 2. Поиск и замена (упрощенная: удаляем все search, добавляем все replace)
            if search_tags_list and current_tag_set.intersection(search_tags_set):
                 tags_after_replace = [t for t in tags if t not in search_tags_set] # Удаляем искомые
                 # Добавляем заменяющие (без дубликатов относительно текущего состояния)
                 existing_after_s_remove = set(tags_after_replace)
                 for add_tag in replace_with_list:
                      if add_tag not in existing_after_s_remove:
                           tags_after_replace.append(add_tag)
                 tags = tags_after_replace
                 current_tag_set = set(tags) # Обновляем set
                 replace_file_count += 1
                 processed = True

            # 3. Добавление активационных тегов (в начало, без дубликатов)
            # Сначала удалим, если они уже есть где-то
            tags_without_act = [t for t in tags if t not in activation_tag_list]
            final_tags = []
            # Затем добавим в начало в обратном порядке
            for act_tag in reversed(activation_tag_list):
                 final_tags.insert(0, act_tag)
            # Добавляем остальные теги
            final_tags.extend(tags_without_act)

            # Проверяем, изменился ли список тегов после добавления активационных
            if len(tags) != len(final_tags) or tags[:len(activation_tag_list)] != activation_tag_list:
                 activation_added_count += 1 # Считаем, даже если тег уже был, но не в начале
                 processed = True
            tags = final_tags # Обновляем основной список тегов


            # 4. Удаление дубликатов (если включено)
            if remove_duplicates:
                seen = set()
                unique_tags = []
                # Сохраняем порядок активационных тегов
                act_tags_part = tags[:len(activation_tag_list)]
                other_tags_part = tags[len(activation_tag_list):]

                unique_tags.extend(act_tags_part) # Активационные всегда уникальны (по идее)
                seen.update(act_tags_part)

                for tag in other_tags_part:
                    if tag not in seen:
                        unique_tags.append(tag)
                        seen.add(tag)
                if len(tags) != len(unique_tags):
                     tags = unique_tags
                     processed = True

            # 5. Сортировка (если включено, после добавления активационных и удаления дублей)
            if sort_alpha:
                 # Сортируем все кроме активационных тегов
                 act_tags_part = tags[:len(activation_tag_list)]
                 other_tags_part = sorted(tags[len(activation_tag_list):])
                 sorted_tags_final = act_tags_part + other_tags_part

                 if tags != sorted_tags_final: # Проверяем, изменился ли порядок
                      tags = sorted_tags_final
                      processed = True

            # Записываем изменения только если они были
            if processed:
                final_tags_str = ", ".join(tags)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(final_tags_str)

        except Exception as e:
            print(f"  [!] Error curating tags in {txt_file}: {e}", file=sys.stderr)

    print("[+] Tag curation complete.")
    if activation_tag_list: print(f"  Activation tags added/moved to start in {activation_added_count} files.")
    if remove_tags_list: print(f"  Removed {remove_count} instances of specified tags.")
    if search_tags_list: print(f"  Performed search/replace in {replace_file_count} files.")
    if remove_duplicates: print(f"  Duplicate tags removed where found.")
    if sort_alpha: print(f"  Tags sorted alphabetically (after activation tags).")


def prepare_training(paths, args, final_params):
    """
    Проверяет датасет, скачивает модель/VAE, генерирует конфиги.
    Использует final_params для значений, которые могли быть авто-определены.
    """
    print("\n--- Preparing Training ---")

    current_num_repeats = final_params['num_repeats']
    current_batch_size = final_params['train_batch_size']
    current_optimizer = final_params['optimizer']
    current_precision = final_params['precision']

    print("[*] Validating dataset...")
    images_folder = paths["images"]
    total_images = get_image_count(images_folder)
    if total_images == 0:
        print(f"[!] Error: No images found in {images_folder}. Aborting.", file=sys.stderr)
        return None # Возвращаем None при ошибке

    pre_steps_per_epoch = total_images * current_num_repeats
    if current_batch_size <= 0:
         print("[!] Error: Train batch size must be positive.", file=sys.stderr)
         return None
    steps_per_epoch = math.ceil(pre_steps_per_epoch / current_batch_size) if current_batch_size > 0 else 0

    if args.preferred_unit == "Epochs":
        max_train_epochs = args.how_many
        max_train_steps = max_train_epochs * steps_per_epoch
    else:
        max_train_steps = args.how_many
        max_train_epochs = math.ceil(max_train_steps / steps_per_epoch) if steps_per_epoch > 0 else 0

    lr_warmup_steps = int(max_train_steps * args.lr_warmup_ratio) if args.lr_scheduler not in ('constant', 'cosine') else 0

    print(f"  Dataset: {images_folder}")
    print(f"  Image count: {total_images}")
    print(f"  Num repeats: {current_num_repeats} (final)")
    print(f"  Total image steps: {pre_steps_per_epoch}")
    print(f"  Batch size: {current_batch_size} (final)")
    print(f"  Steps per epoch: {steps_per_epoch}")
    if args.preferred_unit == "Epochs":
        print(f"  Target epochs: {max_train_epochs}")
        print(f"  Estimated total steps: {max_train_steps}")
    else:
        print(f"  Target steps: {max_train_steps}")
        print(f"  Estimated epochs: {max_train_epochs:.2f}") # Печатаем как float для точности
    print(f"  Warmup steps: {lr_warmup_steps} ({args.lr_warmup_ratio*100:.1f}%)")
    print(f"  Optimizer: {current_optimizer} (final)")
    print(f"  Precision: {current_precision} (final)")

    if max_train_steps <= 0:
        print("[!] Error: Calculated total training steps is zero or negative.", file=sys.stderr)
        return None

    # Определение URL модели и VAE
    model_url = args.custom_model if args.custom_model else args.base_model
    vae_url = args.custom_vae if args.custom_vae else args.base_vae

    model_file_path = None
    is_diffusers_model = False
    is_local_model = os.path.exists(model_url) # Простой способ проверить локальный путь

    if is_local_model:
         model_file_path = os.path.abspath(model_url)
         print(f"[*] Using local model: {model_file_path}")
         if os.path.isdir(model_file_path):
              if all(os.path.exists(os.path.join(model_file_path, d)) for d in ["unet", "text_encoder", "vae", "scheduler"]):
                   is_diffusers_model = True
                   print("  Detected local Diffusers model format.")
              else:
                   print(f"[!] Error: Local path {model_file_path} is a directory, but doesn't look like a Diffusers model.", file=sys.stderr)
                   return None
    elif 'huggingface.co/' in model_url:
        print(f"[*] Using Hugging Face model: {model_url}")
        if '/blob/' in model_url or '/resolve/' in model_url or model_url.endswith(('.safetensors', '.ckpt')):
             if '/blob/' in model_url: model_url = model_url.replace('/blob/', '/resolve/')
             filename = os.path.basename(model_url.split('?')[0])
             model_file_path = os.path.join(paths['project'], filename)
        else:
             is_diffusers_model = True
             model_file_path = model_url # Передаем ID репозитория
             print("  Assuming Diffusers model format from repository URL.")
    elif 'civitai.com/api/download/models/' in model_url:
        print(f"[*] Using Civitai download URL.")
        match = re.search(r'models/(\d+)', model_url)
        filename = f"civitai_model_{match.group(1)}.safetensors" if match else "civitai_model.safetensors"
        model_file_path = os.path.join(paths['project'], filename)
    else:
        print(f"[!] Error: Unsupported model URL/path format: {model_url}", file=sys.stderr)
        return None

    # Определение VAE
    vae_file_path = None
    is_local_vae = False
    if vae_url: # Проверяем, что vae_url вообще задан
        is_local_vae = os.path.exists(vae_url)

    if is_local_vae:
         vae_file_path = os.path.abspath(vae_url)
         print(f"[*] Using local VAE: {vae_file_path}")
    elif vae_url:
         if 'huggingface.co/' in vae_url:
             if '/blob/' in vae_url or '/resolve/' in vae_url or vae_url.endswith(('.safetensors', '.pt', '.ckpt')):
                 if '/blob/' in vae_url: vae_url = vae_url.replace('/blob/', '/resolve/')
                 filename = os.path.basename(vae_url.split('?')[0])
                 vae_file_path = os.path.join(paths['project'], filename)
             else:
                  vae_file_path = vae_url # ID репозитория
                  print(f"[*] Using Hugging Face VAE repo: {vae_file_path}")
         else: # Другие URL
             filename = os.path.basename(vae_url.split('?')[0]) or "downloaded_vae.safetensors"
             vae_file_path = os.path.join(paths['project'], filename)
             print(f"[*] Using VAE URL: {vae_url}")

    # Загрузка модели и VAE
    aria2c_executable = "aria2c"
    use_aria = False
    try:
        subprocess.run([aria2c_executable, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        use_aria = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass # Просто не используем aria

    # Скачивание модели (если не локальная и не diffusers repo)
    if model_file_path and not is_local_model and not is_diffusers_model:
         if not os.path.exists(model_file_path) or args.force_download_model:
              print(f"[*] Downloading Model to {model_file_path}...")
              downloader = run_cmd if use_aria else requests.get # Упрощенно
              if use_aria:
                  run_cmd([aria2c_executable, model_url, "--console-log-level=warn", "-c", "-x", "8", "-s", "8", "-k", "10M", "-d", os.path.dirname(model_file_path), "-o", os.path.basename(model_file_path)], check=False)
              else:
                  try:
                      import requests
                      print("  (Using requests - may be slow)")
                      response = requests.get(model_url, stream=True, timeout=300)
                      response.raise_for_status()
                      with open(model_file_path, 'wb') as f:
                          for chunk in response.iter_content(chunk_size=1024*1024): f.write(chunk)
                      print("[+] Model download complete.")
                  except Exception as e:
                      print(f"[!] Error downloading model with requests: {e}", file=sys.stderr)
                      if os.path.exists(model_file_path): os.remove(model_file_path)
                      return None
         else:
              print(f"[*] Model file already exists: {model_file_path}")

    # Скачивание VAE (если URL задан, не локальный и не repo ID)
    if vae_url and vae_file_path and not is_local_vae and not ('huggingface.co/' in vae_file_path and not vae_file_path.endswith(('.safetensors', '.pt', '.ckpt'))):
         if not os.path.exists(vae_file_path) or args.force_download_vae:
              print(f"[*] Downloading VAE to {vae_file_path}...")
              if use_aria:
                   run_cmd([aria2c_executable, vae_url, "--console-log-level=warn", "-c", "-x", "8", "-s", "8", "-k", "10M", "-d", os.path.dirname(vae_file_path), "-o", os.path.basename(vae_file_path)], check=False)
              else:
                  try:
                      import requests
                      print("  (Using requests - may be slow)")
                      response = requests.get(vae_url, stream=True, timeout=180)
                      response.raise_for_status()
                      with open(vae_file_path, 'wb') as f:
                          for chunk in response.iter_content(chunk_size=1024*1024): f.write(chunk)
                      print("[+] VAE download complete.")
                  except Exception as e:
                      print(f"[!] Error downloading VAE with requests: {e}", file=sys.stderr)
                      if os.path.exists(vae_file_path): os.remove(vae_file_path)
                      return None
         else:
              print(f"[*] VAE file already exists: {vae_file_path}")
    elif not vae_url:
         print("[*] No external VAE specified.")

    # Генерация конфиг-файлов
    print("[*] Generating configuration files...")
    config_file = os.path.join(paths["config"], f"training_{args.project_name}.toml")
    dataset_config_file = os.path.join(paths["config"], f"dataset_{args.project_name}.toml")

    network_args_list = []
    if args.lora_type.lower() == "locon":
        network_args_list = [f"conv_dim={args.conv_dim}", f"conv_alpha={args.conv_alpha}"]

    mixed_precision_val = "no"
    full_precision_val = False
    if "fp16" in current_precision: mixed_precision_val = "fp16"
    if "bf16" in current_precision: mixed_precision_val = "bf16"
    if "full" in current_precision: full_precision_val = True

    training_dict = {
      "model_arguments": {
          "pretrained_model_name_or_path": model_file_path,
          "vae": vae_file_path if vae_file_path and not is_diffusers_model else None,
          "v_parameterization": args.v_pred,
          # "v_pred_like_loss": args.v_pred, # Убрано, т.к. может быть устаревшим
      },
      "network_arguments": {
          "unet_lr": args.unet_lr,
          "text_encoder_lr": args.text_encoder_lr,
          "network_dim": args.network_dim,
          "network_alpha": args.network_alpha,
          "network_module": "networks.lora",
          "network_args": network_args_list if network_args_list else None,
          "network_train_unet_only": args.text_encoder_lr == 0,
          "network_weights": args.continue_from_lora if args.continue_from_lora else None,
          "scale_weight_norms": None, # Можно добавить как параметр, если нужно
      },
      "optimizer_arguments": {
          "optimizer_type": current_optimizer,
          "learning_rate": args.unet_lr,
          "optimizer_args": final_params['optimizer_args'] if final_params['optimizer_args'] else None,
          "lr_scheduler": args.lr_scheduler,
          "lr_warmup_steps": lr_warmup_steps,
          "lr_scheduler_num_cycles": args.lr_scheduler_num_cycles if args.lr_scheduler == "cosine_with_restarts" else None,
          "lr_scheduler_power": args.lr_scheduler_power if args.lr_scheduler == "polynomial" else None,
          "max_grad_norm": 1.0,
          "loss_type": "l2",
      },
      "dataset_arguments": {
          "cache_latents": args.cache_latents,
          "cache_latents_to_disk": args.cache_latents_to_disk,
          "cache_text_encoder_outputs": args.cache_text_encoder_outputs,
          "keep_tokens": args.keep_tokens,
          "shuffle_caption": args.shuffle_tags and not args.cache_text_encoder_outputs,
          "caption_dropout_rate": args.caption_dropout if args.caption_dropout > 0 else None,
          "caption_tag_dropout_rate": args.tag_dropout if args.tag_dropout > 0 else None,
          "caption_extension": args.caption_extension,
          "color_aug": False, # Можно добавить как параметр
          "face_crop_aug_range": None,
          "random_crop": False,
      },
      "training_arguments": {
          "output_dir": paths["output"],
          "output_name": args.project_name,
          "save_precision": "fp16",
          "save_every_n_epochs": args.save_every_n_epochs if args.save_every_n_epochs and args.save_every_n_epochs > 0 else None,
          "save_last_n_epochs": args.keep_only_last_n_epochs if args.keep_only_last_n_epochs and args.keep_only_last_n_epochs > 0 else None,
          "save_model_as": "safetensors",
          "max_train_epochs": max_train_epochs if args.preferred_unit == "Epochs" else None,
          "max_train_steps": max_train_steps if args.preferred_unit == "Steps" else None,
          "max_data_loader_n_workers": args.max_data_loader_n_workers,
          "persistent_data_loader_workers": True,
          "seed": args.seed,
          "gradient_checkpointing": args.gradient_checkpointing,
          "gradient_accumulation_steps": 1,
          "mixed_precision": mixed_precision_val,
          "full_fp16": full_precision_val if mixed_precision_val == "fp16" else None,
          "full_bf16": full_precision_val if mixed_precision_val == "bf16" else None,
          "logging_dir": paths["logs"],
          "log_prefix": args.project_name,
          "log_with": "tensorboard",
          # "wandb_api_key": args.wandb_key if args.wandb_key else None, # Если раскомментировать wandb
          "lowram": args.lowram,
          "train_batch_size": current_batch_size,
          "xformers": args.cross_attention == "xformers",
          "sdpa": args.cross_attention == "sdpa",
          "noise_offset": args.noise_offset if args.noise_offset > 0 else None,
          "min_snr_gamma": args.min_snr_gamma if args.min_snr_gamma > 0 else None,
          "ip_noise_gamma": args.ip_noise_gamma if args.ip_noise_gamma > 0 else None,
          "multires_noise_iterations": 6 if args.multinoise else None,
          "multires_noise_discount": 0.3 if args.multinoise else None,
          "max_token_length": 225,
          "bucket_reso_steps": args.bucket_reso_steps,
          "min_bucket_reso": args.min_bucket_reso,
          "max_bucket_reso": args.max_bucket_reso,
          "bucket_no_upscale": args.bucket_no_upscale,
          "enable_bucket": True,
          "zero_terminal_snr": args.zero_terminal_snr,
          "min_timestep": 0, # Можно добавить как параметр
          "max_timestep": 1000, # Можно добавить как параметр
      },
      "sample_prompt_arguments": None, # Можно добавить секцию для генерации примеров
    }

    def remove_none_values(d):
        """Recursively remove keys with None values from a dictionary."""
        if isinstance(d, dict):
            return {k: remove_none_values(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            # Фильтруем None из списков, если это необходимо (обычно не нужно для конфигов)
             return [remove_none_values(i) for i in d]
        else:
            return d

    clean_training_dict = remove_none_values(training_dict)

    with open(config_file, "w", encoding='utf-8') as f:
        toml.dump(clean_training_dict, f)
    print(f"  Training config saved to: {config_file}")

    dataset_dict = {
      "general": {
          "resolution": args.resolution,
          "keep_tokens": args.keep_tokens,
          "flip_aug": args.flip_aug,
          "enable_bucket": True,
          "bucket_reso_steps": args.bucket_reso_steps,
          "min_bucket_reso": args.min_bucket_reso,
          "max_bucket_reso": args.max_bucket_reso,
          "bucket_no_upscale": args.bucket_no_upscale,
      },
      "datasets": [
          {
             "subsets": [
                 {
                     "image_dir": images_folder,
                     "num_repeats": current_num_repeats,
                 }
                 # Можно добавить другие параметры subset, если нужно
             ]
          }
      ]
      # TODO: Добавить поддержку регуляризации и нескольких папок
    }

    clean_dataset_dict = remove_none_values(dataset_dict)

    with open(dataset_config_file, "w", encoding='utf-8') as f:
        toml.dump(clean_dataset_dict, f)
    print(f"  Dataset config saved to: {dataset_config_file}")

    return config_file, dataset_config_file


def run_training(paths, args, config_file, dataset_config_file):
    """Запускает процесс тренировки с использованием accelerate."""
    print("\n--- Starting Training ---")
    kohya_dir = paths["kohya"]
    venv_dir = paths["venv"]
    accelerate_executable = os.path.join(venv_dir, 'bin', 'accelerate') if os.path.exists(os.path.join(venv_dir, 'bin')) else os.path.join(venv_dir, 'Scripts', 'accelerate.exe')
    # Предполагаем SDXL как основной, как в ноутбуках
    train_script = os.path.join(kohya_dir, "sdxl_train_network.py")
    if not os.path.exists(train_script):
         train_script = os.path.join(kohya_dir, "train_network.py") # Fallback
         if not os.path.exists(train_script):
              print(f"[!] Error: Training script not found in {kohya_dir}", file=sys.stderr)
              return False

    cmd = [
        accelerate_executable, "launch",
        "--num_cpu_threads_per_process", str(args.num_cpu_threads),
        train_script,
        f"--config_file={config_file}",
        f"--dataset_config={dataset_config_file}"
    ]

    print(f"[*] Launching training command: {' '.join(cmd)}")
    # Запускаем тренировку из папки kohya_ss для правильного разрешения путей в скриптах kohya
    run_cmd(cmd, check=True, cwd=kohya_dir)

    print("\n--- Training Finished ---")
    print(f"[*] LoRA model(s) saved in: {paths['output']}")
    return True


# --- Парсер Аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Local LoRA Training Script with Auto Parameter Detection", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Основные Настройки ---
    g_main = parser.add_argument_group('Main Settings')
    g_main.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    g_main.add_argument("--base_dir", type=str, default=".", help="Base directory for project, kohya_ss, venv.")
    g_main.add_argument("--venv_name", type=str, default="lora_env", help="Virtual environment folder name.")

    # --- Источник Модели ---
    g_model = parser.add_argument_group('Model Source')
    g_model.add_argument("--base_model", type=str, default="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", help="URL or local path to the base model.")
    g_model.add_argument("--custom_model", type=str, default=None, help="Override base_model with a custom URL or local path.")
    g_model.add_argument("--base_vae", type=str, default="https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors", help="URL or local path for the VAE. Leave empty if not needed.")
    g_model.add_argument("--custom_vae", type=str, default=None, help="Override base_vae with a custom URL or local path.")
    g_model.add_argument("--v_pred", action='store_true', help="Set if the base model uses v-prediction.")
    g_model.add_argument("--force_download_model", action='store_true', help="Force download model even if file exists.")
    g_model.add_argument("--force_download_vae", action='store_true', help="Force download VAE even if file exists.")

    # --- Этапы Подготовки Датасета ---
    g_prep = parser.add_argument_group('Dataset Preparation Steps')
    g_prep.add_argument("--skip_scraping", action='store_true', help="Skip Gelbooru scraping step.")
    g_prep.add_argument("--scrape_tags", type=str, default="", help="Tags for Gelbooru scraping (e.g., '1girl solo blue_hair').")
    g_prep.add_argument("--scrape_limit", type=int, default=1000, help="Max images to attempt to fetch from Gelbooru.")
    g_prep.add_argument("--scrape_max_res", type=int, default=3072, help="Max resolution for Gelbooru images.")
    g_prep.add_argument("--scrape_include_parents", action=argparse.BooleanOptionalAction, default=True, help="Include Gelbooru posts with parents.")

    g_prep.add_argument("--skip_deduplication", action='store_true', help="Skip duplicate detection (requires FiftyOne).")
    g_prep.add_argument("--dedup_threshold", type=float, default=0.985, help="Similarity threshold for duplicate detection.")

    g_prep.add_argument("--skip_tagging", action='store_true', help="Skip automatic image tagging step.")
    g_prep.add_argument("--tagging_method", type=str, choices=["wd14", "blip"], default="wd14", help="Tagging method.")
    g_prep.add_argument("--tagger_threshold", type=float, default=0.35, help="Confidence threshold for WD14 Tagger.")
    g_prep.add_argument("--tagger_batch_size", type=int, default=8, help="Batch size for WD14 Tagger.")
    g_prep.add_argument("--blip_min_length", type=int, default=10, help="Min caption length for BLIP.")
    g_prep.add_argument("--blip_max_length", type=int, default=75, help="Max caption length for BLIP.")
    g_prep.add_argument("--tagger_blacklist", type=str, default="bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber", help="Comma-separated tags to remove after WD14 tagging.")
    g_prep.add_argument("--overwrite_tags", action='store_true', help="Overwrite existing tag/caption files during tagging.")

    g_prep.add_argument("--skip_tag_curation", action='store_true', help="Skip tag curation step.")
    g_prep.add_argument("--activation_tag", type=str, default="", help="Activation tag(s) to prepend (comma-separated).")
    g_prep.add_argument("--remove_tags", type=str, default="lowres, bad anatomy, worst quality, low quality", help="Comma-separated tags to remove during curation.")
    g_prep.add_argument("--search_tags", type=str, default="", help="Tags to search for during curation (with --replace_tags).")
    g_prep.add_argument("--replace_tags", type=str, default="", help="Tags to replace found --search_tags with.")
    g_prep.add_argument("--sort_tags_alpha", action='store_true', help="Sort tags alphabetically during curation.")
    g_prep.add_argument("--remove_duplicate_tags", action='store_true', help="Remove duplicate tags within each caption file.")

    # --- Настройки Тренировки ---
    g_train = parser.add_argument_group('Training Settings')
    g_train.add_argument("--resolution", type=int, default=1024, help="Training resolution.")
    g_train.add_argument("--caption_extension", type=str, default=".txt", help="Caption file extension.")
    g_train.add_argument("--shuffle_tags", action=argparse.BooleanOptionalAction, default=True, help="Shuffle caption tags (disabled if caching text embeddings).")
    g_train.add_argument("--keep_tokens", type=int, default=1, help="Number of tokens to keep at caption start.")
    g_train.add_argument("--flip_aug", action='store_true', help="Enable flip augmentation.")
    g_train.add_argument("--num_repeats", type=int, default=None, metavar='N', help="Repeats per image (overrides --auto_repeats).")
    g_train.add_argument("--auto_repeats", action='store_true', help="Auto-determine repeats based on image count.")
    g_train.add_argument("--preferred_unit", type=str, choices=["Epochs", "Steps"], default="Epochs", help="Unit for training duration.")
    g_train.add_argument("--how_many", type=int, default=10, help="Number of epochs or steps.")
    g_train.add_argument("--save_every_n_epochs", type=int, default=1, metavar='N', help="Save checkpoint every N epochs (0=only last).")
    g_train.add_argument("--keep_only_last_n_epochs", type=int, default=10, metavar='N', help="Keep only the last N saved epochs (0=keep all).")
    g_train.add_argument("--caption_dropout", type=float, default=0.0, metavar='RATE', help="Caption dropout rate (0-1).")
    g_train.add_argument("--tag_dropout", type=float, default=0.0, metavar='RATE', help="Tag dropout rate (0-1).")

    # --- Параметры Обучения ---
    g_learn = parser.add_argument_group('Learning Parameters')
    g_learn.add_argument("--unet_lr", type=float, default=3e-4, help="U-Net learning rate.")
    g_learn.add_argument("--text_encoder_lr", type=float, default=6e-5, help="Text Encoder learning rate (0 to disable).")
    g_learn.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", choices=["constant", "cosine", "cosine_with_restarts", "constant_with_warmup", "linear", "polynomial"], help="Learning rate scheduler.")
    g_learn.add_argument("--lr_scheduler_num_cycles", type=int, default=3, help="Cycles for cosine_with_restarts.")
    g_learn.add_argument("--lr_scheduler_power", type=float, default=1.0, help="Power for polynomial scheduler.")
    g_learn.add_argument("--lr_warmup_ratio", type=float, default=0.05, help="LR warmup ratio (0.0-0.2).")
    g_learn.add_argument("--min_snr_gamma", type=float, default=8.0, help="Min SNR gamma (<= 0 to disable).")
    g_learn.add_argument("--noise_offset", type=float, default=0.0, help="Noise offset (0 to disable).")
    g_learn.add_argument("--ip_noise_gamma", type=float, default=0.0, help="Instance Prompt Noise Gamma (SDXL specific, 0 to disable).")
    g_learn.add_argument("--multinoise", action='store_true', help="Enable multiresolution noise.")
    g_learn.add_argument("--zero_terminal_snr", action='store_true', help="Enable zero terminal SNR.")

    # --- Структура LoRA ---
    g_lora = parser.add_argument_group('LoRA Structure')
    g_lora.add_argument("--lora_type", type=str, choices=["LoRA", "LoCon"], default="LoRA", help="Type of LoRA network.")
    g_lora.add_argument("--network_dim", type=int, default=32, help="Network dimension (rank).")
    g_lora.add_argument("--network_alpha", type=int, default=16, help="Network alpha.")
    g_lora.add_argument("--conv_dim", type=int, default=16, help="Conv dimension for LoCon.")
    g_lora.add_argument("--conv_alpha", type=int, default=8, help="Conv alpha for LoCon.")
    g_lora.add_argument("--continue_from_lora", type=str, default=None, metavar='PATH', help="Path to existing LoRA file to continue training.")

    # --- Параметры Тренировки (Технические) ---
    g_tech = parser.add_argument_group('Technical Training Parameters')
    g_tech.add_argument("--auto_vram_params", action='store_true', help="Auto-set batch_size, optimizer, precision based on VRAM.")
    g_tech.add_argument("--train_batch_size", type=int, default=None, metavar='N', help="Batch size (overrides auto).")
    g_tech.add_argument("--cross_attention", type=str, choices=["sdpa", "xformers"], default="sdpa", help="Cross attention implementation.")
    g_tech.add_argument("--precision", type=str, choices=["float", "fp16", "bf16", "full_fp16", "full_bf16"], default=None, metavar='TYPE', help="Training precision (overrides auto).")
    g_tech.add_argument("--cache_latents", action=argparse.BooleanOptionalAction, default=True, help="Cache latents.")
    g_tech.add_argument("--cache_latents_to_disk", action=argparse.BooleanOptionalAction, default=True, help="Cache latents to disk.")
    g_tech.add_argument("--cache_text_encoder_outputs", action='store_true', help="Cache text encoder outputs (disables TE training/shuffle).")
    g_tech.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing.")
    g_tech.add_argument("--optimizer", type=str, default=None, choices=["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor"], metavar='OPT', help="Optimizer algorithm (overrides auto).")
    g_tech.add_argument("--use_recommended_optimizer_args", action='store_true', help="Use recommended args for selected optimizer.")
    g_tech.add_argument("--optimizer_args", type=str, default="", help="Additional optimizer arguments (e.g., 'weight_decay=0.01 betas=[0.9,0.99]').")
    g_tech.add_argument("--max_data_loader_n_workers", type=int, default=2, help="Workers for data loading.")
    g_tech.add_argument("--seed", type=int, default=42, help="Random seed.")
    g_tech.add_argument("--num_cpu_threads", type=int, default=2, help="CPU threads per process for Accelerate.")
    g_tech.add_argument("--lowram", action='store_true', help="Enable Kohya low RAM optimizations.")

    # --- Настройки Бакетов ---
    g_bucket = parser.add_argument_group('Bucket Settings')
    g_bucket.add_argument("--bucket_reso_steps", type=int, default=64, help="Steps for bucket resolution.")
    g_bucket.add_argument("--min_bucket_reso", type=int, default=256, help="Minimum bucket resolution.")
    g_bucket.add_argument("--max_bucket_reso", type=int, default=4096, help="Maximum bucket resolution.")
    g_bucket.add_argument("--bucket_no_upscale", action='store_true', help="Disable upscaling aspect ratio buckets.")

    # --- Режим Запуска ---
    g_run = parser.add_argument_group('Run Mode')
    g_run.add_argument("--run_all", action='store_true', help="Run all steps: setup, prep, train.")
    g_run.add_argument("--run_setup_only", action='store_true', help="Only run environment setup and dependency installation.")
    g_run.add_argument("--run_prep_only", action='store_true', help="Run setup and dataset preparation steps only.")
    g_run.add_argument("--run_train_only", action='store_true', help="Run training step only (assumes setup and prep done).")

    return parser.parse_args()


# --- Основная Логика ---
def main():
    args = parse_arguments()

    # Определение режима запуска
    run_setup = args.run_all or args.run_setup_only or args.run_prep_only or not (args.run_setup_only or args.run_prep_only or args.run_train_only)
    run_prep = args.run_all or args.run_prep_only
    run_train = args.run_all or args.run_train_only
    run_train_only = args.run_train_only # Эта переменная используется для проверки

    if run_train_only and (args.run_setup_only or args.run_prep_only):
         print("[!] Error: Cannot combine --run_train_only with setup/prep only modes.", file=sys.stderr)
         sys.exit(1)
    if args.run_setup_only and args.run_prep_only:
         print("[!] Error: Cannot use --run_setup_only and --run_prep_only together.", file=sys.stderr)
         sys.exit(1)

    print("--- LoRA Training Script ---")
    print(f"Project: {args.project_name}")
    print(f"Base Directory: {os.path.abspath(args.base_dir)}")

    # --- 1. Настройка Окружения ---
    # Эта функция теперь также устанавливает зависимости, если venv активен
    paths = setup_environment(args.base_dir, args.project_name, args.venv_name)
    gpu_vram = get_gpu_vram() # Определяем VRAM после возможной установки torch

    if args.run_setup_only:
        print("\n--- Setup Complete ---")
        sys.exit(0)

    # --- Авто-определение параметров ---
    final_params = {}
    auto_vram_settings = {}
    if args.auto_vram_params:
        auto_vram_settings = determine_vram_parameters(gpu_vram)
        print(f"[*] Auto VRAM params: {auto_vram_settings}")

    final_params['train_batch_size'] = args.train_batch_size if args.train_batch_size is not None else auto_vram_settings.get('batch_size', 1)
    final_params['optimizer'] = args.optimizer if args.optimizer is not None else auto_vram_settings.get('optimizer', 'AdamW8bit')
    final_params['precision'] = args.precision if args.precision is not None else auto_vram_settings.get('precision', 'fp16')
    final_params['optimizer_args'] = []

    if args.use_recommended_optimizer_args:
         optimizer_lower = final_params['optimizer'].lower()
         if optimizer_lower == "adamw8bit": final_params['optimizer_args'] = ["weight_decay=0.1", "betas=[0.9,0.99]"]
         elif optimizer_lower == "prodigy": final_params['optimizer_args'] = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
         # ... (добавить другие)
    elif args.optimizer_args:
         final_params['optimizer_args'] = args.optimizer_args.split(' ')

    # Num repeats (считаем картинки перед подготовкой)
    image_count_initial = get_image_count(paths['images'])
    auto_repeats_val = 0
    if args.auto_repeats:
        auto_repeats_val = determine_num_repeats(image_count_initial)
        print(f"[*] Auto num_repeats initial: {auto_repeats_val} (based on {image_count_initial} images)")
    final_params['num_repeats'] = args.num_repeats if args.num_repeats is not None else auto_repeats_val if args.auto_repeats else 10

    print("--- Final Effective Parameters (Initial) ---")
    print(f"  Batch Size: {final_params['train_batch_size']}")
    print(f"  Optimizer: {final_params['optimizer']}")
    print(f"  Optimizer Args: {' '.join(final_params['optimizer_args']) if final_params['optimizer_args'] else 'None'}")
    print(f"  Precision: {final_params['precision']}")
    print(f"  Num Repeats: {final_params['num_repeats']}")
    print("------------------------------------------")

    # --- 2. Подготовка Датасета ---
    config_file = None
    dataset_config_file = None

    if run_prep:
        print("\n--- Starting Dataset Preparation ---")
        if not args.skip_scraping: scrape_images(args.scrape_tags, paths["images"], paths["config"], args.project_name, args.scrape_max_res, args.scrape_include_parents, args.scrape_limit)
        if not args.skip_deduplication: detect_duplicates(paths["images"], args.dedup_threshold)
        if not args.skip_tagging: tag_images(paths["images"], paths["kohya"], args.tagging_method, args.tagger_batch_size, args.tagger_threshold, args.caption_extension, args.tagger_blacklist, args.blip_min_length, args.blip_max_length, args.overwrite_tags)
        if not args.skip_tag_curation: curate_tags(paths["images"], args.activation_tag, args.remove_tags, args.search_tags, args.replace_tags, args.caption_extension, args.sort_tags_alpha, args.remove_duplicate_tags)
        print("\n--- Dataset Preparation Complete ---")

        # Пересчитываем количество картинок *после* подготовки
        image_count_final = get_image_count(paths['images'])
        if args.auto_repeats and args.num_repeats is None: # Обновляем repeats, если они авто и не заданы явно
            final_params['num_repeats'] = determine_num_repeats(image_count_final)
            print(f"[*] Auto num_repeats updated: {final_params['num_repeats']} (based on {image_count_final} final images)")
            print("--- Final Effective Parameters (Updated after Prep) ---")
            print(f"  Num Repeats: {final_params['num_repeats']}")
            print("----------------------------------------------------")


        if args.run_prep_only:
             print("[*] Generating config files after preparation...")
             prep_result = prepare_training(paths, args, final_params)
             if not prep_result: sys.exit(1)
             print("\n--- Preparation Complete (including config generation) ---")
             sys.exit(0)

    # --- 3. Подготовка к Тренировке (Конфиги, Модель) ---
    # Генерация конфигов происходит здесь, если не был prep_only
    if run_train or (run_prep and not args.run_prep_only): # Если был run_prep, но не prep_only
         prep_result = prepare_training(paths, args, final_params)
         if not prep_result:
              print("[!] Error during training preparation. Aborting.", file=sys.stderr)
              sys.exit(1)
         config_file, dataset_config_file = prep_result
    elif run_train_only:
         # Ищем существующие конфиги
         config_file = os.path.join(paths["config"], f"training_{args.project_name}.toml")
         dataset_config_file = os.path.join(paths["config"], f"dataset_{args.project_name}.toml")
         if not os.path.exists(config_file) or not os.path.exists(dataset_config_file):
              print(f"[!] Error: Config files not found for --run_train_only mode.", file=sys.stderr)
              print(f"  Expected: {config_file}, {dataset_config_file}")
              sys.exit(1)
         print(f"[*] Using existing config files for training: {config_file}, {dataset_config_file}")


    # --- 4. Запуск Тренировки ---
    if run_train:
        # Убедимся, что конфиг файлы были определены
        if config_file and dataset_config_file:
            run_training(paths, args, config_file, dataset_config_file)
        else:
            print("[!] Error: Config files were not generated or found. Cannot start training.", file=sys.stderr)
            sys.exit(1)
    else:
         print("\n--- Skipping Training Step ---")

    print("\n--- Script Finished ---")

# --- Точка входа ---
if __name__ == "__main__":
    main()