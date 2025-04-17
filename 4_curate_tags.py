# -*- coding: utf-8 -*-
import os
import argparse
import sys
# Импорт общих утилит
try:
    import common_utils
except ImportError:
    print("[X] CRITICAL ERROR: common_utils.py not found.", file=sys.stderr); sys.exit(1)

# --- Функция курирования ---
def curate_tags(images_folder, activation_tag, remove_tags, search_tags, replace_tags, caption_ext=".txt", sort_alpha=False, remove_duplicates=False):
    """Обрабатывает файлы тегов: добавляет активационный тег, удаляет ненужные, ищет/заменяет."""
    print("\n--- Tag Curation ---")
    if not os.path.isdir(images_folder): print(f"[!] Image directory not found: {images_folder}. Skipping.", file=sys.stderr); return

    try:
        tag_files = [f for f in os.listdir(images_folder) if f.lower().endswith(caption_ext)]
        if not tag_files: print(f"[*] No tag files ('{caption_ext}') found to curate in {images_folder}."); return
    except OSError as e: print(f"[!] Error accessing image directory {images_folder}: {e}. Skipping.", file=sys.stderr); return

    # Используем split_tags из common_utils
    activation_tag_list = common_utils.split_tags(activation_tag)
    remove_tags_set = set(common_utils.split_tags(remove_tags))
    search_tags_set = set(common_utils.split_tags(search_tags))
    replace_with_list = common_utils.split_tags(replace_tags)

    remove_count = 0; replace_file_count = 0; activation_added_count = 0; processed_files_count = 0

    print(f"[*] Curating tags in {len(tag_files)} files...")
    for txt_file in tag_files:
        filepath = os.path.join(images_folder, txt_file)
        processed = False
        try:
            with open(filepath, 'r', encoding='utf-8') as f: current_tags_str = f.read()
            tags = common_utils.split_tags(current_tags_str); original_tags_tuple = tuple(tags)

            # 1. Удаление тегов
            if remove_tags_set:
                tags_after_remove = [t for t in tags if t not in remove_tags_set]
                removed_count_this_file = len(tags) - len(tags_after_remove)
                if removed_count_this_file > 0: remove_count += removed_count_this_file; tags = tags_after_remove; processed = True

            # 2. Поиск и замена
            current_tag_set = set(tags)
            if search_tags_set and current_tag_set.intersection(search_tags_set):
                 tags_after_replace = [t for t in tags if t not in search_tags_set]
                 existing_after_s_remove = set(tags_after_replace)
                 for add_tag in replace_with_list:
                      if add_tag not in existing_after_s_remove: tags_after_replace.append(add_tag)
                 tags = tags_after_replace; replace_file_count += 1; processed = True

            # 3. Активационные теги
            if activation_tag_list:
                tags_without_act = [t for t in tags if t not in activation_tag_list]
                final_tags = activation_tag_list + tags_without_act
                if tuple(tags) != tuple(final_tags): activation_added_count +=1; processed = True
                tags = final_tags

            # 4. Удаление дубликатов
            if remove_duplicates:
                seen = set(); unique_tags = []
                num_act_tags = len(activation_tag_list) if activation_tag_list else 0
                act_tags_part = tags[:num_act_tags]; other_tags_part = tags[num_act_tags:]
                unique_tags.extend(act_tags_part); seen.update(act_tags_part)
                for tag in other_tags_part:
                    if tag not in seen: unique_tags.append(tag); seen.add(tag)
                if len(tags) != len(unique_tags): tags = unique_tags; processed = True

            # 5. Сортировка
            if sort_alpha:
                 num_act_tags = len(activation_tag_list) if activation_tag_list else 0
                 act_tags_part = tags[:num_act_tags]; other_tags_part = sorted(tags[num_act_tags:])
                 sorted_tags_final = act_tags_part + other_tags_part
                 if tags != sorted_tags_final: tags = sorted_tags_final; processed = True

            # Запись изменений
            if processed:
                final_tags_str = ", ".join(tags)
                if final_tags_str or original_tags_tuple:
                    with open(filepath, 'w', encoding='utf-8') as f: f.write(final_tags_str)
                    processed_files_count += 1
                else: print(f"  [*] Skipping write for {txt_file} as it became empty.")

        except Exception as e: print(f"  [!] Error curating tags in {txt_file}: {e}", file=sys.stderr)

    print(f"[+] Tag curation complete. {processed_files_count} files were modified.")
    if activation_tag_list: print(f"  Activation tags added/moved: {activation_added_count} files.")
    if remove_tags_set: print(f"  Removed {remove_count} instances of specified tags.")
    if search_tags_set: print(f"  Performed search/replace in {replace_file_count} files.")
    if remove_duplicates: print(f"  Duplicate tags removed where found.")
    if sort_alpha: print(f"  Tags sorted alphabetically (after activation tags).")

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 4: Curate existing tag files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing project folder.")
    parser.add_argument("--caption-extension", type=str, default=".txt", help="Extension of tag/caption files.")
    parser.add_argument("--activation-tag", type=str, default="", help="Activation tag(s) to prepend (comma-separated).")
    parser.add_argument("--remove-tags", type=str, default="", help="Comma-separated tags to remove.")
    parser.add_argument("--search-tags", type=str, default="", help="Comma-separated tags to search for (with --replace-tags).")
    parser.add_argument("--replace-tags", type=str, default="", help="Comma-separated tags to replace found --search-tags with.")
    parser.add_argument("--sort-tags-alpha", action='store_true', help="Sort tags alphabetically (after activation tags).")
    parser.add_argument("--remove-duplicate-tags", action='store_true', help="Remove duplicate tags within each file.")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")

    print("--- Step 4: Curate Tags ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Image Folder: {images_folder}")

    curate_tags(
        images_folder, args.activation_tag, args.remove_tags, args.search_tags,
        args.replace_tags, args.caption_extension, args.sort_tags_alpha,
        args.remove_duplicate_tags
    )
    print("\n--- Step 4 Finished ---")