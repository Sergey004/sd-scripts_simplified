# -*- coding: utf-8 -*-
import os
import argparse
import time
import sys
# Импорты fiftyone и sklearn обернуты в try-except
try:
    import fiftyone as fo
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[!] Error: fiftyone or numpy or scikit-learn not found.", file=sys.stderr)
    print("[-] Please ensure the venv is active and dependencies were installed via setup_environment.py.", file=sys.stderr)
    fo = None # Устанавливаем в None, чтобы проверки ниже сработали

# --- Константы ---
SUPPORTED_IMG_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --- Функция дедупликации ---
def detect_duplicates(images_folder, threshold=0.985):
    """Обнаруживает дубликаты с помощью FiftyOne."""
    print("\n--- Duplicate Detection ---")
    if fo is None: # Проверяем импорт
        print("[!] FiftyOne library not available. Skipping duplicate detection.")
        return

    if not os.path.isdir(images_folder):
        print(f"[!] Image directory not found: {images_folder}. Skipping.", file=sys.stderr)
        return

    try:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(SUPPORTED_IMG_TYPES)]
        if len(image_files) < 2:
            print("[!] Less than 2 images found. Skipping duplicate detection.")
            return
    except OSError as e:
        print(f"[!] Error accessing image directory {images_folder}: {e}. Skipping.", file=sys.stderr)
        return

    dataset = None
    dataset_name = f"dedupe_check_{os.path.basename(images_folder)}_{int(time.time())}"
    try:
        # Удаляем старый датасет с таким именем
        if fo.dataset_exists(dataset_name):
            print(f"[*] Deleting existing temporary dataset: {dataset_name}")
            fo.delete_dataset(dataset_name)

        print(f"[*] Creating temporary FiftyOne dataset: {dataset_name}")
        # Используем from_images для простоты
        dataset = fo.Dataset.from_images(image_files, name=dataset_name, persistent=False)

        print("[*] Computing embeddings (may download CLIP model)...")
        model = fo.zoo.load_zoo_model("clip-vit-base32-torch")
        embeddings = dataset.compute_embeddings(model, batch_size=32)

        print("[*] Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)

        print(f"[*] Finding duplicates with threshold > {threshold}...")
        id_map = [s.id for s in dataset.select_fields(["id"])]
        samples_to_tag_delete = set()
        samples_to_tag_duplicate = set()

        for idx, sample in enumerate(dataset):
            if sample.id not in samples_to_tag_delete:
                dup_indices = np.where(similarity_matrix[idx] > threshold)[0]
                has_valid_duplicates = False
                for dup_idx in dup_indices:
                    dup_id = id_map[dup_idx]
                    if dup_id != sample.id and dup_id not in samples_to_tag_delete:
                        samples_to_tag_delete.add(dup_id)
                        has_valid_duplicates = True
                if has_valid_duplicates:
                    samples_to_tag_duplicate.add(sample.id)

        # Применяем теги
        delete_count = 0; duplicate_count = 0
        if samples_to_tag_delete or samples_to_tag_duplicate:
            print("[*] Applying tags to dataset...")
            with fo.ProgressBar() as pb:
                 for sample in pb(dataset):
                    tagged = False
                    if sample.id in samples_to_tag_delete:
                        sample.tags.append("delete"); delete_count += 1; tagged = True
                    elif sample.id in samples_to_tag_duplicate:
                        sample.tags.append("has_duplicates"); duplicate_count += 1; tagged = True
                    if tagged: sample.save()
        else:
            print("[*] No duplicates found above the threshold.")

        if delete_count > 0 or duplicate_count > 0:
            print(f"[+] Marked {delete_count} images with 'delete' tag.")
            print(f"[+] Marked {duplicate_count} images (kept) with 'has_duplicates' tag.")
            print("[!] IMPORTANT: This script only TAGS duplicates. MANUALLY delete files for images tagged 'delete'.")

    except ImportError:
         print("[!] FiftyOne or scikit-learn import failed again. Skipping.", file=sys.stderr)
    except Exception as e:
        print(f"[!] An error occurred during duplicate detection: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
         # Удаление временного датасета
         if dataset and fo.dataset_exists(dataset.name):
             try: fo.delete_dataset(dataset.name); print(f"[*] Deleted temporary FiftyOne dataset '{dataset.name}'.")
             except Exception as e_del: print(f"[!] Error deleting temporary dataset '{dataset.name}': {e_del}", file=sys.stderr)

# --- Парсер аргументов ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Step 2: Detect duplicate images using FiftyOne.")
    parser.add_argument("--project_name", type=str, required=True, help="Name of the project.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing project folder.")
    parser.add_argument("--dedup_threshold", type=float, default=0.985, help="Similarity threshold for duplicates (0-1).")
    return parser.parse_args()

# --- Точка входа ---
if __name__ == "__main__":
    args = parse_arguments()
    base_dir = os.path.abspath(args.base_dir)
    project_dir = os.path.join(base_dir, args.project_name)
    images_folder = os.path.join(project_dir, "dataset")

    print("--- Step 2: Detect Duplicates ---")
    print(f"[*] Project: {args.project_name}")
    print(f"[*] Image Folder: {images_folder}")
    print(f"[*] Threshold: {args.dedup_threshold}")

    detect_duplicates(images_folder, args.dedup_threshold)
    print("\n--- Step 2 Finished ---")