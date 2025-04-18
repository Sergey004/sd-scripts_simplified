# Активируйте venv перед запуском!
# Например: source ./Loras/lora_env/bin/activate

python master_train.py \
    --project_name "Lora_name" \
    --base_dir "./Loras" \
    --run_steps "tag,curate,config,train" \
\
    --base_model "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors" \
    --base_vae "stabilityai/sdxl-vae" \
\
    --tagging_method "wd14" \
    --tagger_threshold 0.35 \
    --tagger_batch_size 8 \
    --tagger_blacklist "bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber, parody, style parody, official alternate costume, official alternate hairstyle, official alternate hair length, alternate costume, alternate hairstyle, alternate hair length, alternate hair color" \
\
    --caption_extension ".txt" \
    --activation_tag "put_yours" \
    --remove_tags "candy, musical note, gradient, white background, background, green eyes, heart, gradient background, solo, artist name, traditional media, multicolored background, checkered background, purple background, looking at viewer, simple background, male focus, brown eyes, feet out of frame, underwear only, window, sitting, couch, night sky, night, starry sky, brown fur" \
    --remove_duplicate_tags \
\
    --resolution 1024 \
    --keep_tokens 1 \
    --preferred_unit "Epochs" \
    --how_many 10 \
    --save_every_n_epochs 1 \
    --keep_only_last_n_epochs 10 \
    --unet_lr 3e-4 \
    --text_encoder_lr 6e-5 \
    --lr_scheduler "cosine_with_restarts" \
    --lr_scheduler_num_cycles 3 \
    --lr_warmup_ratio 0.05 \
    --min_snr_gamma 8.0 \
    --multinoise \
    --lora_type "LoRA" \
    --network_dim 32 \
    --network_alpha 16 \
    --auto_vram_params \
    --auto_repeats \
    --cross_attention "sdpa" \
    --cache_latents \
    --cache_latents_to_disk \
    --gradient_checkpointing \
    --use_recommended_optimizer_args \
    --seed 42 \
    --lowram \
    --max_bucket_reso 4096 \
    --num_cpu_threads 12