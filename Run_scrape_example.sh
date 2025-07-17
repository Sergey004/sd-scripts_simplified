#!/bin/bash
# Script to scrape artworks from a specific author using master_train.py (gallery-dl backend)

# --- Activate environment if needed ---
# source ./Loras/lora_env/bin/activate

echo "Starting image scraping..."
echo "Using Python version: $(python --version)"

python master_train.py \
    --project_name "Lora_name" \
    --base_dir "./Loras" \
    --run_steps "scrape" \
    --base_model "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors" \
    --base_vae "stabilityai/sdxl-vae" \
    --source "furaffinity" \
    --user "or charactor" \
    --scrape_limit 250 \
    --cookies "www.furaffinity.net_cookies.txt" 

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Image scraping finished successfully."
else
    echo "Image scraping failed with exit code $EXIT_CODE."
fi
