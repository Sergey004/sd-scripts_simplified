# Need rewrite


# Local LoRA Training Pipeline (SDXL Focused)

This set of scripts provides a modular pipeline for training LoRA (Low-Rank Adaptation) models for Stable Diffusion (with a focus on SDXL), designed to run on a local machine (Linux/WSL recommended, may require adaptation for Windows/macOS). It is based on the workflow presented in the Colab notebooks by Hollowstrawberry and utilizes `kohya_ss/sd-scripts` as the training backend.

## Features

* **Modular Structure:** The entire process is broken down into separate scripts for each stage (setup, scraping, deduplication, tagging, curation, config generation, training).
* **Master Script (`master_train.py`):** Manages the execution of selected pipeline stages using a unified set of parameters.
* **Automatic Tagging:** Supports WD14 Tagger and BLIP for automatically generating captions/tags for images.
* **Tag Curation:** Provides options for adding activation tags, removing unwanted tags, search/replace, removing duplicates, and sorting.
* **Duplicate Detection:** Integrates with FiftyOne (if installed) to detect and tag visually similar images.
* **Automatic Parameter Detection:** Optionally auto-determines some training parameters (`batch_size`, `optimizer`, `precision`, `num_repeats`) based on GPU VRAM and image count.
* **Flexible Stage Control:** Run the entire pipeline, preparation only, training only, or any combination of steps via command-line arguments.
* **Control Pauses:** Optional pauses between stages and an initial pause for manual dataset preparation.
* **Isolated Environment:** Uses a Python virtual environment (venv) to manage dependencies.

## Prerequisites

1. **Python:** Version 3.10 or 3.11 recommended (for `kohya_ss` compatibility).
2. **Git:** Required for cloning `kohya_ss/sd-scripts`.
3. **CUDA:** Installed NVIDIA drivers and CUDA Toolkit compatible with the PyTorch version specified in `0_setup_environment.py` (default is `cu124` for PyTorch 2.5.1). Check compatibility on the [PyTorch website](https://pytorch.org/).
4. **System:** Scripts were primarily tested under Linux/WSL. Running on native Windows or macOS might require additional setup (especially regarding paths and certain dependencies).
5. **(Optional but recommended) `aria2c`:** Utility for fast parallel downloads (`sudo apt install aria2` or equivalent). Downloads will be slower without it.
6. **(Optional) `FiftyOne`:** Required for the duplicate detection step (`2_detect_duplicates.py`). It's installed by `setup_environment.py`, but might require additional database setup on non-Ubuntu systems (see FiftyOne documentation).

## Installation and Setup

The installation process is performed **once** using the `setup_environment.py` script.

1. **Download or clone** all the scripts (`0_...py` - `6_...py`, `master_train.py`, `common_utils.py`) into a single folder.
2. **Open a terminal** in that folder.
3. **Run the setup script:**

    ```bash
    python setup_environment.py --base_dir "./Loras" --venv_name "lora_env" --kohya_dir_name "kohya_ss"
    ```

    * `--base_dir`: Specify the directory where project folders, the venv, and kohya_ss will be created (default is the current folder `.`). Using `./Loras` is recommended.
    * `--venv_name`: Name for the virtual environment directory (default `lora_env`).
    * `--kohya_dir_name`: Name for the `kohya_ss/sd-scripts` directory (default `kohya_ss`).

    This script will:
    * Create the virtual environment (if it doesn't exist).
    * Install *all* necessary Python packages (including PyTorch, xformers, kohya, fiftyone, etc.) *inside* this environment.
    * Clone the `kohya_ss/sd-scripts` repository and check out a specific stable commit.
    * Apply necessary patches to the `kohya_ss` scripts.

4. **Activate the virtual environment:** After `0_setup_environment.py` finishes successfully, you **MUST** activate the created environment **BEFORE** running `master_train.py` or other scripts (1-6). The activation commands will be printed at the end of the setup script:
    * **Linux/macOS:** `source ./Loras/lora_env/bin/activate`
    * **Windows CMD:** `.\Loras\lora_env\Scripts\activate`
    * **Windows PowerShell:** `.\Loras\lora_env\Scripts\Activate.ps1`
    (Replace `./Loras` and `lora_env` with your values if you changed them). You should see the environment name (e.g., `(lora_env)`) at the beginning of your terminal prompt.

## Usage

The primary way to run the pipeline is via `master_train.py`.

**General Command Format:**

```bash
python master_train.py --project_name <your_project_name> --base_model <path_or_URL> [stage_options] [stage_parameters...]
```

**Before Running:**

* **Activate the venv!**
* **Prepare Dataset:** Place your images in the `<base_dir>/<project_name>/dataset/` folder. If you have pre-existing tag files (`.txt`), place them there as well, ensuring filenames match the corresponding images. The master script will pause after creating directories (unless `--skip_initial_pause` is used) to allow you time for this.

**Example Usage:**

1. **Full Run (all steps) with a local model and auto-params:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "/path/to/my/model.safetensors" \
        --base_vae "/path/to/my/vae.safetensors" \
        --skip_scrape \
        --skip_deduplication \
        --tagging_method "wd14" \
        --activation_tag "mycoollora style" \
        --remove_tags "simple background, text" \
        --remove_duplicate_tags \
        --auto_vram_params \
        --auto_repeats \
        # ... other training parameters if needed ...
    ```

    * This will create folders, pause for dataset check, then run tagging, curation, config generation, and training, pausing between major steps (unless `--no_wait` is added).

2. **Run only Tagging, Curation, Config Generation, and Training:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "/path/to/my/model.safetensors" \
        --run_steps "tag,curate,config,train" \
        # ... provide all parameters needed for steps tag, curate, config, train ...
    ```

3. **Run Training Only (configs must already exist):**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "/path/to/my/model.safetensors" \
        --run_steps "train"
    ```

    * Note: `--base_model` is still required by the `master_train.py` parser, even though it's not directly used by the `6_run_training.py` script itself (which reads the path from the config file).

4. **Full run without any pauses:**

    ```bash
    # Activate venv first!
    python master_train.py \
        --project_name "MyCoolLora" \
        --base_dir "./Loras" \
        --base_model "/path/to/my/model.safetensors" \
        --no_wait \
        --skip_initial_pause \
        # ... other parameters ...
    ```

**Key `master_train.py` Arguments:**

* `--project_name`: Your project's name (required).
* `--base_dir`: The base directory (default `.`).
* `--base_model`: Path or URL to the base model (required).
* `--run_steps`: Specifies which steps to run (default is all). Use comma-separated names (e.g., `tag,curate,train`).
* `--skip_steps`: Specifies steps to skip (e.g., `scrape,dedupe`). Overrides `--run_steps`.
* `--no_wait`: Disables the "Press Enter" pause between steps.
* `--skip_initial_pause`: Skips the initial pause intended for dataset preparation.
* Other arguments are passed to the relevant child scripts. Use `python master_train.py --help` to see all available options.

## Running Individual Scripts

For debugging or finer control, you can run the individual scripts (`1_...py` - `6_...py`) separately (after activating the venv). Each script has its own specific set of arguments, viewable with `--help`:

```bash
# Activate venv first!
python 1_scrape_images.py --help
python 3_tag_images.py --help
python 5_generate_configs.py --help
# etc.
```

## Configuration Files

The `5_generate_configs.py` script creates two files in the `<base_dir>/<project_name>/config/` directory:

* `training_<project_name>.toml`: Contains training parameters (LR, optimizer, steps, paths, etc.).
* `dataset_<project_name>.toml`: Contains dataset parameters (resolution, bucketing, repeats, image folder paths).

These files are then used by `6_run_training.py` to launch the `kohya_ss` trainer via `accelerate launch`. Advanced users can edit these `.toml` files directly before running step 6.

## Troubleshooting

* **`ImportError` / `ModuleNotFoundError`:** Ensure you have **activated the correct virtual environment** (`lora_env`) created by `0_setup_environment.py`. Run `python 0_setup_environment.py ...` again if you suspect installation issues. Make sure `common_utils.py` is in the same directory as the other scripts.
* **`command not found` (e.g., `git`, `aria2c`, `python`):** Ensure the necessary system utilities are installed and available in your system's `PATH`.
* **CUDA / PyTorch Errors:** Verify compatibility between your NVIDIA driver version, CUDA Toolkit version, and the PyTorch version specified in `0_setup_environment.py`. Consider installing a different PyTorch/CUDA version by modifying the constants in `0_setup_environment.py` and re-running it.
* **`FileNotFoundError` for `.toml` config files:** Ensure Step 5 (`5_generate_configs.py`) completed successfully before running Step 6 (`6_run_training.py`).
* **Insufficient VRAM / Out-of-Memory Errors:** Decrease `--train_batch_size` (potentially to 1), use `--precision fp16`, ensure `--gradient_checkpointing` is enabled (default), use `--optimizer AdamW8bit`, enable `--lowram`.

## Acknowledgements

This project builds upon the work and code from the following authors and projects:

* [Hollowstrawberry](https://github.com/hollowstrawberry/kohya-colab) for the original Colab notebooks and workflow.
* [kohya-ss](https://github.com/kohya-ss/sd-scripts) for the core LoRA training toolkit.
* Google Gemini 2.5 Pro for creating this
