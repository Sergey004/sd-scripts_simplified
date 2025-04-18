# -*- coding: utf-8 -*-
import os
import argparse
import sys

# --- Функция для замены тегов ---
def replace_tags_with_single(images_folder, common_tag, caption_ext=".txt"):
    """
    Заменяет содержимое всех файлов тегов в папке одним общим тегом.

    Args:
        images_folder (str): Путь к папке с файлами тегов.
        common_tag (str): Единственный тег, который нужно оставить.
        caption_ext (str): Расширение файлов тегов (например, ".txt").
    """
    print(f"\n--- Replacing Tags in {images_folder} ---")
    if not os.path.isdir(images_folder):
        print(f"[!] Error: Directory not found: {images_folder}", file=sys.stderr)
        return

    if not common_tag:
        print("[!] Error: Common tag cannot be empty.", file=sys.stderr)
        return

    modified_count = 0
    error_count = 0
    processed_count = 0

    try:
        all_files = os.listdir(images_folder)
    except OSError as e:
        print(f"[!] Error accessing directory {images_folder}: {e}", file=sys.stderr)
        return

    tag_files = [f for f in all_files if f.lower().endswith(caption_ext)]

    if not tag_files:
        print(f"[*] No tag files with extension '{caption_ext}' found.")
        return

    print(f"[*] Found {len(tag_files)} tag files. Replacing content with: '{common_tag}'")

    for filename in tag_files:
        processed_count += 1
        filepath = os.path.join(images_folder, filename)
        try:
            # Читаем текущее содержимое для сравнения (опционально, но полезно)
            # current_content = ""
            # with open(filepath, 'r', encoding='utf-8') as f_read:
            #     current_content = f_read.read().strip()

            # Перезаписываем файл новым тегом
            # Простая перезапись эффективнее, чем проверка содержимого
            with open(filepath, 'w', encoding='utf-8') as f_write:
                f_write.write(common_tag)
            modified_count += 1
            # if current_content != common_tag: # Если нужно считать только реально измененные
            #    modified_count += 1

        except Exception as e:
            print(f"  [!] Error processing file {filename}: {e}", file=sys.stderr)
            error_count += 1

    print(f"[+] Finished replacing tags.")
    print(f"  Total files processed: {processed_count}")
    print(f"  Files overwritten: {modified_count}")
    if error_count > 0:
        print(f"  Errors encountered: {error_count}")

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Replace content of all tag files in a directory with a single common tag.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("images_folder", help="Path to the directory containing the tag files.")
    parser.add_argument("common_tag", help="The single tag to write into every file.")
    parser.add_argument("--caption-ext", type=str, default=".txt", help="Extension of the tag files to process.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()

    # Преобразуем путь к абсолютному для надежности
    images_folder_abs = os.path.abspath(args.images_folder)

    print(f"[*] Target Folder: {images_folder_abs}")
    print(f"[*] Common Tag: '{args.common_tag}'")
    print(f"[*] File Extension: '{args.caption_ext}'")

    replace_tags_with_single(images_folder_abs, args.common_tag, args.caption_ext)

    print("\n--- Script Finished ---")