# Local LoRA Training Pipeline (SDXL and Beyond)

This set of scripts provides a modular pipeline for training LoRA (Low-Rank Adaptation) models for Stable Diffusion, with a focus on **SDXL**, **Illustrious**, and **Pony Diffusion** models. Designed to run on a local machine (Linux/WSL recommended, may require adaptation for Windows/macOS), it is based on the workflow from Hollowstrawberry's Colab notebooks and utilizes `kohya_ss/sd-scripts` as the training backend. Updated for 2025 with support for the **REX** learning rate scheduler, **CAME** optimizer, and predefined training profiles.

## Features

* **Modular Structure:** The pipeline is divided into separate scripts for each stage (setup, scraping, deduplication, tagging, curation, config generation, training).
* **Master Script (`master_train.py`):** Orchestrates pipeline stages with unified parameters and supports predefined profiles for **Illustrious** and **Pony Diffusion**.
* **Training Profiles:** Preconfigured settings for **Character**, **Style**, and **Concept** training on **Illustrious** and **Pony Diffusion** models, optimizing parameters like learning rates, optimizer, and scheduler.
* **REX Scheduler:** Implements the **Reflected Exponential** learning rate scheduler for improved training stability, particularly for **Illustrious** models.
* **CAME Optimizer:** Supports the **Clipped Adaptive Momentum Estimation** optimizer, optimized for **Illustrious** training.
* **Automatic Tagging:** Uses WD14 Tagger or BLIP to generate captions/tags for images.
* **Tag Curation:** Options to add activation tags, remove unwanted tags, search/replace, remove duplicates, and sort tags.
* **Duplicate Detection:** Integrates with FiftyOne (if installed) to detect and tag visually similar images.
* **Automatic Parameter Detection:** Auto-determines `batch_size`, `optimizer`, `precision`, and `num_repeats` based on GPU VRAM and image count.
* **Flexible Stage Control:** Run the entire pipeline, specific steps (e.g., preparation or training), or any combination via command-line arguments.
* **Control Pauses:** Optional pauses between stages and an initial pause for manual dataset preparation.
* **Isolated Environment:** Uses a Python virtual environment (venv) to manage dependencies.

## Prerequisites

1. **Python:** Version 3.10 or 3.11 recommended (for `kohya_ss` compatibility).
2. **Git:** Required for cloning `kohya_ss/sd-scripts`.
3. **CUDA:** Installed NVIDIA drivers and CUDA Toolkit compatible with the PyTorch version specified in `0_setup_environment.py` (default is `cu124` for PyTorch 2.5.1). Check compatibility on the [PyTorch website](https://pytorch.org/).
4. **System:** Scripts were primarily tested under Linux/WSL. Running on native Windows or macOS may require path adjustments and additional setup.
5. **(Optional but recommended) `aria2c`:** Utility for fast parallel downloads (`sudo apt install aria2` or equivalent). Downloads will be slower without it.
6. **(Optional) `FiftyOne`:** Required for duplicate detection (`2_detect_duplicates.py`). Installed by `setup_environment.py`, but may require additional database setup on non-Ubuntu systems (see [FiftyOne documentation](https://voxel51.com/docs/fiftyone/)).

## Installation and Setup

The installation process is performed **once** using the `setup_environment.py` script.

1. **Download or clone** all the scripts (`0_...py` - `6_...py`, `master_train.py`, `common_utils.py`, `rex_lr.py`) into a single folder.
2. **Open a terminal** in that folder.
3. **Run the setup script:**

    ```bash
    python setup_environment.py --base_dir "./Loras" --venv_name "lora_env" --kohya_dir_name "kohya_ss"
    ```

    * `--base_dir`: Directory for project folders, venv, and kohya_ss (default `.`). `./Loras` is recommended.
    * `--venv_name`: Name for the virtual environment directory (default `lora_env`).
    * `--kohya_dir_name`: Name for the `kohya_ss/sd-scripts` directory (default `kohya_ss`).

    This script will:
    * Create the virtual environment (if it doesn't exist).
    * Install Python packages (PyTorch, xformers, kohya_ss, fiftyone, CAME optimizer, etc.) inside the venv.
    * Clone the `kohya_ss/sd-scripts` repository and check out a stable commit.
    * Apply necessary patches to `kohya_ss` scripts.

4. **Activate the virtual environment:** After `0_setup_environment.py` completes, activate the environment **before** running `master_train.py` or other scripts:
    * **Linux/macOS:** `source ./Loras/lora_env/bin/activate`
    * **Windows CMD:** `.\Loras\lora_env\Scripts\activate`
    * **Windows PowerShell:** `.\Loras\lora_env\Scripts\Activate.ps1`
    Replace `./Loras` and `lora_env` with your values if changed. The terminal prompt should show the environment name (e.g., `(lora_env)`).

## Usage

The primary way to run the pipeline is via `master_train.py`, which supports predefined profiles for **Illustrious** and **Pony Diffusion** models.

**General Command Format:**

```bash
python master_train.py --project_name <your_project_name> --base_model <path_or_URL> --profile <profile_name> [stage_options] [stage_parameters...]
```

**Before Running:**

* **Activate the venv!**
* **Prepare Dataset:** Place images in `<base_dir>/<project_name>/dataset/`. Include pre-existing tag files (`.txt`) with matching filenames if available. The script pauses after creating directories (unless `--skip_initial_pause` is used) for dataset preparation.

### Training Profiles

The pipeline includes predefined profiles to simplify training for **Illustrious** and **Pony Diffusion** models. Specify a profile using `--profile <profile_name>`.

| Profile                | Model       | Optimizer | Scheduler | UNET LR | TE LR   | Dim | Alpha | Steps/Epochs | Keep Tokens | Flip Aug |
|-----------------------|-------------|-----------|-----------|---------|---------|-----|-------|--------------|-------------|----------|
| illustrious_character | Illustrious | CAME      | REX       | 6e-5    | 6e-6    | 64  | 32    | 10 epochs    | 1           | No       |
| illustrious_style     | Illustrious | CAME      | REX       | 0.0002  | 0.00002 | 128 | 64    | 2500 steps   | 0           | No       |
| illustrious_concept   | Illustrious | CAME      | REX       | 0.0005  | 0.00005 | 32  | 16    | 10 epochs    | 1           | Yes      |
| pony_character        | Pony Diff.  | AdaFactor | Cosine    | 0.0001  | 0.00001 | 16  | 8     | 3600 steps   | 1           | No       |
| pony_style            | Pony Diff.  | Prodigy   | Cosine    | 0.75    | 0.75    | 24  | 24    | 3600 steps   | 0           | No       |
| pony_concept          | Pony Diff.  | AdaFactor | Cosine    | 0.0001  | 0.00001 | 32  | 16    | 3600 steps   | 1           | Yes      |

**Note:** Profiles override default parameters (e.g., `unet_lr`, `optimizer`, `lr_scheduler`). You can further customize settings via command-line arguments.

### Example Usage

1. **Full Run with Illustrious Character Profile:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "illustrious-xl-v0.1" \
        --base_vae "stabilityai/sdxl-vae" \
        --profile illustrious_character \
        --scrape_tags "1girl, blue_hair" \
        --scrape_limit 40 \
        --activation_tag "mycoollora" \
        --remove_tags "lowres, bad_anatomy" \
        --auto_vram_params \
        --auto_repeats
    ```

    This creates folders, pauses for dataset check, scrapes images, tags, curates, generates configs, and trains using the `illustrious_character` profile with **CAME** and **REX**.

2. **Run Tagging, Curation, Config, and Training with Pony Style:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
        --profile pony_style \
        --run_steps "tag,curate,config,train" \
        --tagging_method "wd14" \
        --activation_tag "mystyle" \
        --remove_tags "text, watermark"
    ```

3. **Training Only (Existing Configs):**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "illustrious-xl-v0.1" \
        --run_steps "train"
    ```

    **Note:** `--base_model` is required by `master_train.py`, even if `6_run_training.py` reads it from the config file.

4. **Full Run without Pauses:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "stabilityai/stable-diffusion-xl-base-1.0" \
        --profile pony_character \
        --no_wait \
        --skip_initial_pause \
        --scrape_tags "1boy, knight" \
        --scrape_limit 20 \
        --activation_tag "myknight"
    ```

### Key `master_train.py` Arguments

* `--project_name`: Your project's name (required).
* `--base_dir`: Base directory (default `.`).
* `--base_model`: Path or URL to the base model (required, e.g., `illustrious-xl-v0.1` or `stabilityai/stable-diffusion-xl-base-1.0`).
* `--base_vae`: Path or URL to an external VAE (e.g., `stabilityai/sdxl-vae`).
* `--profile`: Training profile (e.g., `illustrious_character`, `pony_style`).
* `--run_steps`: Steps to run (e.g., `tag,curate,train`; default is all: `setup,scrape,dedupe,tag,curate,config,train`).
* `--skip_steps`: Steps to skip (e.g., `scrape,dedupe`).
* `--no_wait`: Disables "Press Enter" pauses between steps.
* `--skip_initial_pause`: Skips the initial dataset preparation pause.
* `--lr_scheduler`: Learning rate scheduler (`rex`, `cosine`, etc.; default `cosine_with_restarts`).
* `--optimizer`: Optimizer (`CAME`, `Prodigy`, `AdamW8bit`, etc.; auto-detected if `--auto_vram_params` is used).
* Other arguments are passed to child scripts. Run `python master_train.py --help` for a full list.

### Using REX Scheduler

The **REX (Reflected Exponential)** scheduler, implemented in `rex_lr.py`, provides stable learning rate adjustments, ideal for **Illustrious** models. It is automatically enabled in `illustrious_*` profiles or can be set manually:

```bash
--lr_scheduler rex --unet_lr 6e-5 --text_encoder_lr 6e-6
```

### Using CAME Optimizer

The **CAME (Clipped Adaptive Momentum Estimation)** optimizer is optimized for **Illustrious** models and included in `illustrious_*` profiles. Manual configuration:

```bash
--optimizer CAME --optimizer_args "weight_decay=0.01 betas=[0.9,0.999]"
```

**Note:** Ensure the CAME optimizer is installed via `setup_environment.py`. If unavailable, replace with `Prodigy` or `AdamW8bit`.

## Running Individual Scripts

For debugging or finer control, run individual scripts (`1_...py` - `6_...py`) after activating the venv. View arguments with `--help`:

```bash
# Activate venv first!
python 1_scrape_images.py --help
python 3_tag_images.py --help
python 5_generate_configs.py --help
```

## Configuration Files

The `5_generate_configs.py` script generates two files in `<base_dir>/<project_name>/config/`:

* `training_<project_name>.toml`: Training parameters (learning rates, optimizer, scheduler, steps, paths, etc.).
* `dataset_<project_name>.toml`: Dataset parameters (resolution, bucketing, repeats, image paths).

Edit these `.toml` files directly for advanced customization before running `6_run_training.py`.

## Dataset Recommendations (2025)

* **Image Count:** 10-40 images (10 for Character, up to 40 for Concept).
* **Resolution:** 1024 pixels (up to 4096 offline).
* **Tagging:** Use booru-style tags. Include activation tags (e.g., `mycoollora`) for new concepts.
* **Preparation:** Crop images, ensure diverse poses/angles, remove low-quality images.

## Training Recommendations (2025)

* **Base Models:**
  * **Illustrious**: Use `illustrious-xl-v0.1` with `stabilityai/sdxl-vae`.
  * **Pony Diffusion**: Use SDXL-based models (e.g., `stabilityai/stable-diffusion-xl-base-1.0`).
* **Batch Size:** 2-4 (adjust based on VRAM; use `--auto_vram_params`).
* **Steps/Epochs:**
  * **Illustrious**: 10-20 epochs.
  * **Pony Diffusion**: >3600 steps.
* **VRAM:** 16+ GB recommended for Illustrious. Use `--lowram` for lower VRAM GPUs.
* **Offline Training:** Consider Colab, Runpod, or VastAI for resource-intensive tasks.

## Troubleshooting

* **`ImportError` / `ModuleNotFoundError`:** Ensure the venv is activated and `common_utils.py` and `rex_lr.py` are in the script directory. Re-run `0_setup_environment.py` if issues persist.
* **`REX not found`:** Verify `rex_lr.py` exists and is correctly imported in `common_utils.py`.
* **`CAME not found`:** Check if `came-optimizer` is installed in `setup_environment.py`. If unavailable, use `Prodigy` or `AdamW8bit`.
* **`command not found` (e.g., `git`, `aria2c`):** Install missing system utilities and ensure they are in your `PATH`.
* **CUDA / PyTorch Errors:** Confirm compatibility between NVIDIA drivers, CUDA Toolkit, and PyTorch version in `0_setup_environment.py`. Adjust PyTorch/CUDA versions if needed.
* **`FileNotFoundError` for `.toml` files:** Ensure `5_generate_configs.py` ran successfully before `6_run_training.py`.
* **Insufficient VRAM / Out-of-Memory Errors:** Reduce `--train_batch_size` (e.g., to 1), use `--precision bf16` or `--precision fp16`, enable `--gradient_checkpointing` (default), use `--optimizer AdamW8bit`, or enable `--lowram`.
* **Overtraining:** Lower `--unet_lr` or use intermediate checkpoints (`--save_every_n_epochs`).

## Acknowledgements

This project builds upon the work and code from the following authors and projects:

* [Hollowstrawberry](https://github.com/hollowstrawberry/kohya-colab) for the original Colab notebooks and workflow.
* [kohya-ss](https://github.com/kohya-ss/sd-scripts) for the core LoRA training toolkit.
* [REX: Revisiting Budgeted Training with an Improved Schedule](https://arxiv.org/abs/2107.04197) for the REX scheduler inspiration.
* [Civitai](https://civitai.com/) for training guides and model resources.
* Google Gemini 2.5 Pro and for assistance in creating the original pipeline.
* Gork 3 for updates

## Sources

* [Opinionated Guide to All LoRA Training 2025 Update](https://civitai.com/articles/1716/opinionated-guide-to-all-lora-training-2025-update)
* [Illustrious LoRA Training Discussion](https://civitai.com/articles/9148/illustrious-lora-training-discussion)

---

### Changes Made

1. **Added REX Scheduler:**
   * Included description of the **REX** scheduler in the Features section and Usage section.
   * Noted its implementation in `rex_lr.py` and integration with `kohya_ss/sd-scripts`.
   * Added troubleshooting for REX-related issues.

2. **Added CAME Optimizer:**
   * Added **CAME** to the Features section and Usage section, noting its optimization for **Illustrious** models.
   * Included installation details in the Setup section (via `setup_environment.py`).
   * Added troubleshooting for CAME-related issues.

3. **Introduced Training Profiles:**
   * Added a table of predefined profiles for **Illustrious** and **Pony Diffusion** in the Usage section.
   * Explained how profiles override default parameters and can be specified via `--profile`.
   * Updated example commands to use profiles.

4. **Updated Recommendations for 2025:**
   * Added Dataset Recommendations and Training Recommendations sections based on the provided `.md` file, tailored for **Illustrious** and **Pony Diffusion**.
   * Included specific advice on image count, resolution, batch size, steps/epochs, and VRAM.

5. **Enhanced Example Commands:**
   * Updated examples to include `--profile` and demonstrate usage with **Illustrious** and **Pony Diffusion** models.
   * Added examples for manual REX and CAME configuration.

6. **Improved Troubleshooting:**
   * Added specific troubleshooting for REX and CAME issues.
   * Expanded VRAM and overtraining solutions to align with 2025 recommendations.

7. **Updated Acknowledgements and Sources:**
   * Added references to the REX paper and Civitai articles.
   * Retained original acknowledgements for Hollowstrawberry and kohya-ss.

8. **Maintained Original Structure:**
   * Preserved the modular structure, tone, and formatting of the original README.
   * Ensured new content integrates seamlessly with existing sections.

This updated README provides a comprehensive guide for users, incorporating the new features while maintaining clarity and compatibility with the existing pipeline. If you need further tweaks or additional details, let me know!
