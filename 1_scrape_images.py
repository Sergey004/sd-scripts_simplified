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

from bs4 import BeautifulSoup

# --- Функция скачивания с Gelbooru ---



# --- Функция скачивания с FurAffinity ---

# --- Универсальная функция скачивания через gallery-dl ---
def scrape_images_gallery_dl(url, images_folder, limit=1000, extractor_opts=None, cookies_file=None):
    """Скачивает изображения с любого поддерживаемого сайта через gallery-dl по ссылке. Поддержка cookies.txt."""
    print(f"\n--- Image Scraping (gallery-dl) ---\n[*] URL: {url}")
    import subprocess
    import shlex
    # Build gallery-dl command
    cmd = [
        "gallery-dl",
        url,
        "-d", images_folder,
        "-v",              # Enable verbose logging
        "-r", "",          # No delay between requests
        "-o", "proxy-env=false", # Disable proxy from env
        "--config-ignore"  # Ignore config for max speed (user request)
    ]
    # gallery-dl does not support a universal --limit argument for all sites.
    # Try to use --range 1-N for all sites if limit is set and > 0
    if limit and limit > 0:
        cmd += ["--range", f"1-{limit}"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
        print(f"[*] Using cookies: {cookies_file}")
    # Run gallery-dl
    try:
        print(f"[>] Running: {' '.join(shlex.quote(x) for x in cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Stream gallery-dl output live
        for line in process.stdout:
            print(f"[gallery-dl] {line.rstrip()}")
        process.wait()
        if process.returncode != 0:
            print(f"[!] gallery-dl exited with code {process.returncode}")
            return
    except FileNotFoundError:
        print("[!] gallery-dl is not installed or not in PATH. Please install it with 'pip install gallery-dl' or via your package manager.")
        return
    except Exception as e:
        print(f"[!] gallery-dl error: {e}")
        return

    # Move all downloaded files from subfolders to images_folder
    import glob
    import shutil
    print("[i] Moving all images from subfolders to dataset root...")
    for root, dirs, files in os.walk(images_folder):
        if root == images_folder:
            continue
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(images_folder, file)
            # Avoid overwrite
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(images_folder, f"{base}_{i}{ext}")):
                    i += 1
                dst = os.path.join(images_folder, f"{base}_{i}{ext}")
            shutil.move(src, dst)
    # Optionally remove empty subfolders
    for root, dirs, files in os.walk(images_folder, topdown=False):
        if root == images_folder:
            continue
        if not os.listdir(root):
            os.rmdir(root)
    print(f"[+] All images moved to: {images_folder}")

# --- Функция-обёртка для поддержки разных сайтов ---
def scrape_images_supported_site(site, tags, images_folder, config_folder, project_name, limit=1000, user=None, cookies_file=None):
    # Gelbooru: только теги
    if site == "gelbooru":
        if tags:
            tags_str = tags.replace(" ", "+")
            url = f"https://gelbooru.com/index.php?page=post&s=list&tags={tags_str}"
            print(f"[*] Gelbooru: tag search: {tags}")
            extractor_opts = None
        else:
            print("[!] No tags specified for gelbooru.")
            return
    # Instagram: поддержка профилей, тегов, постов и др.
    if site == "instagram":
        # Возможности: Avatars, Collections, Followers, Followed Users, Guides, Highlights, User Profile Information, Posts, Reels, Saved Posts, Stories, Tag Searches, Tagged Posts, User Profiles
        # Примеры ссылок: https://www.instagram.com/{user}/, https://www.instagram.com/explore/tags/{tag}/
        if user:
            url = f"https://www.instagram.com/{user}/"
            print(f"[*] Instagram: user profile: {user}")
            print("[i] Instagram supports: Avatars, Collections, Followers, Followed Users, Guides, Highlights, User Profile Information, Posts, Reels, Saved Posts, Stories, Tag Searches, Tagged Posts, User Profiles.")
            extractor_opts = None
        elif tags:
            tag = tags.split()[0]
            url = f"https://www.instagram.com/explore/tags/{tag}/"
            print(f"[*] Instagram: tag search: {tag}")
            print("[i] Instagram supports: Avatars, Collections, Followers, Followed Users, Guides, Highlights, User Profile Information, Posts, Reels, Saved Posts, Stories, Tag Searches, Tagged Posts, User Profiles.")
            extractor_opts = None
        else:
            print("[!] Instagram requires --user or --scrape-tags argument.")
            return
    # e621: tags only
    if site == "e621":
        if tags:
            tags_str = "+".join(tags.split())
            url = f"https://e621.net/posts?tags={tags_str}"
            print(f"[*] e621: tag search: {tags}")
            extractor_opts = None
        else:
            print("[!] No tags specified for e621.")
            return
    """Скачивание с любого поддерживаемого gallery-dl сайта по ссылке, тегам или пользователю. Поддержка cookies.txt."""
    # FurAffinity: user -> gallery, tags -> search
    if site == "furaffinity":
        if user:
            url = f"https://www.furaffinity.net/gallery/{user}/"
            print(f"[*] Using user gallery: {user}")
            extractor_opts = {"furaffinity": {"all": True}}
        elif tags:
            tags_str = "+".join(tags.split())
            url = f"https://www.furaffinity.net/search/?q={tags_str}"
            print(f"[*] Using tag search: {tags}")
            extractor_opts = {"furaffinity": {"all": True}}
        else:
            print("[!] Neither tags nor user specified.")
            return
    # DeviantArt: user -> gallery, tags -> search
    elif site == "deviantart":
        if user:
            url = f"https://www.deviantart.com/{user}/gallery/all"
            print(f"[*] DeviantArt: user gallery: {user}")
            extractor_opts = None
        elif tags:
            tags_str = "+".join(tags.split())
            url = f"https://www.deviantart.com/tag/{tags_str}"
            print(f"[*] DeviantArt: tag search: {tags}")
            extractor_opts = None
        else:
            print("[!] Neither tags nor user specified.")
            return
    # ArtStation: user only
    elif site == "artstation":
        if user:
            url = f"https://www.artstation.com/{user}/projects"
            print(f"[*] ArtStation: user projects: {user}")
            extractor_opts = None
        else:
            print("[!] ArtStation requires --user argument.")
            return
    # Pixiv: user -> artworks, tags -> search
    elif site == "pixiv":
        if user:
            url = f"https://www.pixiv.net/en/users/{user}/artworks"
            print(f"[*] Pixiv: user gallery: {user}")
            extractor_opts = None
        elif tags:
            tags_str = "%20".join(tags.split())
            url = f"https://www.pixiv.net/en/tags/{tags_str}/artworks"
            print(f"[*] Pixiv: tag search: {tags}")
            extractor_opts = None
        else:
            print("[!] Neither tags nor user specified.")
            return
    # Generic: if url is passed directly
    elif site == "custom":
        url = tags
        extractor_opts = None
    else:
        print(f"[!] Unknown site: {site}")
        return
    scrape_images_gallery_dl(url, images_folder, limit=limit, extractor_opts=extractor_opts, cookies_file=cookies_file)

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 1: Scrape images from Gelbooru or supported gallery-dl sites.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing project folder.")
    parser.add_argument("--scrape-tags", type=str, required=False, help="Tags for scraping (e.g., '1girl solo blue_hair').")
    parser.add_argument("--scrape-limit", type=int, default=1000, help="Max images to attempt to fetch.")
    parser.add_argument("--scrape-max-res", type=int, default=3072, help="Max resolution (Gelbooru only, larger images replaced by samples).")
    parser.add_argument("--scrape-include-parents", action=argparse.BooleanOptionalAction, default=True, help="Include posts with parents (Gelbooru only).")
    parser.add_argument(
        "--source",
        type=str,
        choices=["gelbooru", "furaffinity", "deviantart", "artstation", "pixiv", "e621", "instagram", "custom", "all"],
        default="gelbooru",
        help="Image source: gelbooru, furaffinity, deviantart, artstation, pixiv, e621, instagram, custom, or all (all = search character on all sites). For Instagram, you can use --user for profile or --scrape-tags for tag search. Supported: Avatars, Collections, Followers, Followed Users, Guides, Highlights, User Profile Information, Posts, Reels, Saved Posts, Stories, Tag Searches, Tagged Posts, User Profiles."
    )
    parser.add_argument("--user", type=str, required=False, help="Username/author for gallery download (universal for all supported gallery-dl sites).")
    parser.add_argument("--cookies", type=str, required=False, help="Path to cookies.txt for gallery-dl (if site requires authentication).")
    parser.add_argument("--type", type=str, choices=["character", "author"], default="character", help="Type of search: 'character' (by tags) or 'author' (by user/username).")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")
    config_folder = os.path.join(project_dir, "config")

    print("--- Step 1: Scrape Images ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Image Folder: {images_folder}")
    print(f"[*] Config Folder: {config_folder}")
    print(f"[*] Tags: {args.scrape_tags}")
    print(f"[*] Source: {args.source}")
    if args.cookies:
        print(f"[*] Cookies file: {args.cookies}")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(config_folder, exist_ok=True)

    if args.type == "character":
        if args.source == "gelbooru":
            scrape_images_supported_site(
                "gelbooru",
                args.scrape_tags if args.scrape_tags else None,
                images_folder,
                config_folder,
                args.project_name,
                args.scrape_limit,
                user=None,
                cookies_file=args.cookies if args.cookies else None
            )
        elif args.source == "all":
            # Поиск персонажа по всем поддерживаемым сайтам (кроме авторов)
            for site in ["gelbooru", "furaffinity", "deviantart", "artstation", "pixiv", "e621", "instagram"]:
                print(f"\n[=] Scraping from {site}...")
                scrape_images_supported_site(
                    site,
                    args.scrape_tags if args.scrape_tags else None,
                    images_folder,
                    config_folder,
                    args.project_name,
                    args.scrape_limit,
                    user=None,
                    cookies_file=args.cookies if args.cookies else None
                )
        elif args.source in ["furaffinity", "deviantart", "artstation", "pixiv", "custom", "e621", "instagram"]:
            scrape_images_supported_site(
                args.source,
                args.scrape_tags if args.scrape_tags else None,
                images_folder,
                config_folder,
                args.project_name,
                args.scrape_limit,
                user=None,
                cookies_file=args.cookies if args.cookies else None
            )
    elif args.type == "author":
        if args.source == "gelbooru":
            print("[!] Author search is not supported for Gelbooru.")
        elif args.source == "all":
            # Поиск автора по всем поддерживаемым сайтам (где есть user)
            for site in ["furaffinity", "deviantart", "artstation", "pixiv"]:
                print(f"\n[=] Scraping from {site} (author)...")
                scrape_images_supported_site(
                    site,
                    None,
                    images_folder,
                    config_folder,
                    args.project_name,
                    args.scrape_limit,
                    user=args.user if args.user else None,
                    cookies_file=args.cookies if args.cookies else None
                )
        elif args.source in ["furaffinity", "deviantart", "artstation", "pixiv"]:
            scrape_images_supported_site(
                args.source,
                None,
                images_folder,
                config_folder,
                args.project_name,
                args.scrape_limit,
                user=args.user if args.user else None,
                cookies_file=args.cookies if args.cookies else None
            )
        else:
            print(f"[!] Author search is not supported for source: {args.source}")
    print("\n--- Step 1 Finished ---")