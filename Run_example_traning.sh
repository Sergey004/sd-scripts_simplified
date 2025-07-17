#!/bin/bash
# (Необязательно, но хорошая практика)

# --- Активация окружения ---
# Убедитесь, что путь к вашему venv правильный!
# source ./Loras/lora_env/bin/activate || exit 1 # Пример с выходом при ошибке

echo "Starting LoRA Style Training Pipeline..."
echo "Ensure your virtual environment is activated."
echo "Using Python version: $(python --version)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python master_train.py \
    --project_name "Lora_name" \
    --base_dir "./Loras" \
    --run_steps "tag,curate,config,train" \
\
    --base_model "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors" \
    --base_vae "stabilityai/sdxl-vae" \
\
    --overwrite_tags \
    --tagging_method "wd14" \
    --tagger_threshold 0.45 \
    --tagger_batch_size 16 \
    --tagger_blacklist "nsfw, lowres, bad anatomy, signature, watermark, text, artist name, virtual youtuber, parody, poorly drawn, amateur drawing, sketch" \
    --caption_extension ".txt" \
    --activation_tag "put_yours" \
    --remove_tags "solo, duo, group, simple background, white background, grey background, gradient background, abstract background, looking at viewer, upper body, full body, portrait, absurdres, highres, bad quality, normal quality, signature, artist name, text, watermark, logo, copyright, traditional media, scan, photo, official art, fanart, commission, request, meme, joke, parody, nsfw, sfw, child, loli, shota, bear, chubby, fat" \
    --remove_duplicate_tags \
    --sort_tags_alpha \
\
    --resolution 1024 \
    --keep_tokens 1 \
    --preferred_unit "Epochs" \
    --how_many 10 \
    --save_every_n_epochs 1 \
    --keep_only_last_n_epochs 10 \
    --auto_repeats \
\
    --unet_lr 3e-4 \
    --text_encoder_lr 6e-5 \
    --lr_scheduler "cosine_with_restarts" \
    --lr_scheduler_num_cycles 3 \
    --lr_warmup_ratio 0.05 \
    --min_snr_gamma 5.0 \
\
    --lora_type "LoRA" \
    --network_dim 32 \
    --network_alpha 16 \
\
    --train_batch_size 4 \
    --precision "bf16" \
    --cross_attention "sdpa" \
    --cache_latents \
    --auto_vram_params \
    --cache_latents_to_disk \
    --gradient_checkpointing \
    --seed 42 \
    --max_bucket_reso 4096 \
    --num_cpu_threads 12

# --- Проверка завершения ---
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Master script finished successfully."
else
    echo "Master script failed with exit code $EXIT_CODE."
fi