# -*- coding: utf-8 -*-
import os
import argparse
import re
import json
import time
import sys
from urllib.request import Request
from pathlib import Path
# Импорт общих утилит
try:
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr)
    print("[-] Please ensure common_utils.py is in the same directory.", file=sys.stderr)
    sys.exit(1)

# Импортируем requests динамически, если нужно
try:
    import requests
except ImportError:
    requests = None

# --- Функция скачивания ---
def scrape_images(tags, images_folder, config_folder, project_name, max_resolution=3072, include_parents=True, limit=1000):
    """Скачивает изображения с Gelbooru по тегам."""
    print("\n--- Image Scraping (Gelbooru) ---")
    if not tags:
        print("[!] No tags provided (--scrape-tags). Skipping.")
        return

    global requests # Используем глобальную переменную
    if requests is None:
        try:
            print("[*] Trying to import 'requests' library...")
            import requests # Пытаемся импортировать
            print("[+] 'requests' imported successfully.")
        except ImportError:
            print("[!] 'requests' library not found. Please ensure the venv is active and dependencies installed via setup_environment.py.")
            return

    # Проверка доступности aria2c
    aria2c_executable = "aria2c"
    use_aria = False
    try:
        # Используем run_cmd из common_utils
        common_utils.run_cmd([aria2c_executable, "--version"], check=True, capture_output=True)
        use_aria = True
        print("[*] aria2c found, using for faster downloads.")
    except Exception: # Ловим любые ошибки (FileNotFound, CalledProcessError)
        print("[!] aria2c not found or not working. Downloads will be slower via requests.")

    # Код скачивания... (использует SUPPORTED_IMG_TYPES из common_utils)
    tags_str = tags.replace(" ", "+").replace(":", "%3a").replace("&", "%26").replace("(", "%28").replace(")", "%29")
    api_url_template = "https://gelbooru.com/index.php?page=dapi&json=1&s=post&q=index&limit=100&tags={}&pid={}"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    all_image_urls = set(); pid = 0; fetched_count = 0; max_api_limit = 100
    print(f"[*] Fetching image list for tags: {tags} (target limit: {limit})")
    while fetched_count < limit:
        url = api_url_template.format(tags_str, pid); print(f"  Fetching page {pid+1}...")
        try:
            response = requests.get(url, headers={"User-Agent": user_agent}, timeout=45); response.raise_for_status(); data = response.json() # Увеличим таймаут
            if not isinstance(data, dict) or "post" not in data:
                 if isinstance(data, dict) and "@attributes" in data and not data.get("post"): print(f"  No more 'post' entries found on page {pid+1}. Stopping fetch."); break
                 else: print(f"[!] Unexpected API response format on page {pid+1}. Response: {str(data)[:200]}..."); break
            if not data["post"]: print(f"  No more images found on page {pid+1}."); break
            count_on_page = 0
            for post in data["post"]:
                 if not isinstance(post, dict) or not all(k in post for k in ['width', 'height', 'file_url', 'sample_url', 'rating']): print(f"  [!] Skipping post with missing keys: {post.get('id', 'N/A')}"); continue
                 parent_id = post.get('parent_id', 0)
                 try:
                     parent_id = int(parent_id)
                 except (ValueError, TypeError):
                     parent_id = 0
                 if parent_id == 0 or include_parents:
                     img_url = post['file_url'] if post['width'] * post['height'] <= max_resolution**2 else post['sample_url']
                     if img_url and isinstance(img_url, str) and img_url.lower().endswith(common_utils.SUPPORTED_IMG_TYPES):
                        all_image_urls.add(img_url); count_on_page += 1; fetched_count += 1;
                        if fetched_count >= limit: break
            print(f"  Found {count_on_page} valid images on page {pid+1}. Total fetched: {fetched_count}")
            if fetched_count >= limit: break
            if len(data["post"]) < max_api_limit: print("  Reached end of results."); break
            pid += 1; time.sleep(0.3);
        except requests.exceptions.RequestException as e: print(f"[!] Network error fetching page {pid+1}: {e}", file=sys.stderr); time.sleep(3); continue
        except json.JSONDecodeError as e: print(f"[!] Error decoding JSON from page {pid+1}: {e}", file=sys.stderr); print(f"  Response text: {response.text[:200]}..."); break
        except Exception as e: print(f"[!] Unexpected error during fetch loop page {pid+1}: {e}", file=sys.stderr); break
    if not all_image_urls: print("[!] No images found matching criteria."); return
    print(f"[*] Found {len(all_image_urls)} unique image URLs.")
    os.makedirs(config_folder, exist_ok=True); scrape_file = os.path.join(config_folder, f"scrape_{project_name}.txt");
    try:
        with open(scrape_file, "w", encoding='utf-8') as f: f.write("\n".join(sorted(list(all_image_urls))));
        print(f"[*] Image URLs saved to {scrape_file}")
    except OSError as e: print(f"[!] Error saving scrape list {scrape_file}: {e}", file=sys.stderr); return
    print(f"[*] Downloading images to {images_folder}..."); os.makedirs(images_folder, exist_ok=True);
    if use_aria: common_utils.run_cmd([aria2c_executable, "--console-log-level=warn", "--async-dns=false", "-c", "-x", "8", "-s", "8", "-k", "10M", "-d", images_folder, "-i", scrape_file], check=False)
    else:
        # ... (requests download loop как раньше) ...
        for i, img_url in enumerate(sorted(list(all_image_urls))):
            try:
                try:
                    filename = os.path.basename(Request(img_url).full_url.split('?')[0].split('/')[-1])
                    if not filename:
                        raise ValueError("Empty filename")
                except Exception:
                    filename = f"image_{i+1}{Path(img_url.split('?')[0]).suffix or '.jpg'}"
                filepath = os.path.join(images_folder, filename);
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0: continue
                print(f"  Downloading {i+1}/{len(all_image_urls)}: {filename} from {img_url[:50]}...")
                img_response = requests.get(img_url, headers={"User-Agent": user_agent}, stream=True, timeout=120); img_response.raise_for_status(); # Увеличен таймаут
                with open(filepath, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=1024*1024): f.write(chunk);
                time.sleep(0.1);
            except requests.exceptions.RequestException as e: print(f"  [!] Network error downloading {filename or img_url[:50]}: {e}", file=sys.stderr)
            except Exception as e: print(f"  [!] Unexpected error downloading {filename or img_url[:50]}: {e}", file=sys.stderr)

    downloaded_count = common_utils.get_image_count(images_folder) # Используем утилиту
    print(f"[+] Download attempt complete. Total images in dataset folder: {downloaded_count}")

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 1: Scrape images from Gelbooru.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Используем дефисы в именах аргументов для командной строки
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing project folder.")
    parser.add_argument("--scrape-tags", type=str, required=True, help="Tags for Gelbooru scraping (e.g., '1girl solo blue_hair').")
    parser.add_argument("--scrape-limit", type=int, default=1000, help="Max images to attempt to fetch.")
    parser.add_argument("--scrape-max-res", type=int, default=3072, help="Max resolution (larger images replaced by samples).")
    parser.add_argument("--scrape-include-parents", action=argparse.BooleanOptionalAction, default=True, help="Include posts with parents.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    # Доступ к аргументам через подчеркивание
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")
    config_folder = os.path.join(project_dir, "config")

    print("--- Step 1: Scrape Images ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Image Folder: {images_folder}")
    print(f"[*] Config Folder: {config_folder}")
    print(f"[*] Tags: {args.scrape_tags}")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(config_folder, exist_ok=True)

    scrape_images(
        args.scrape_tags,
        images_folder,
        config_folder,
        args.project_name,
        args.scrape_max_res,
        args.scrape_include_parents,
        args.scrape_limit
    )
    print("\n--- Step 1 Finished ---")