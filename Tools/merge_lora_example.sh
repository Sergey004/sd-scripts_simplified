python ../Loras/kohya_ss/networks/merge_lora.py \
--save_precision bf16 \
--save_to /path/to/merge \
--models \
    lora1.safetensors \
    lora2.safetensors \
--ratios 0.5 0.5
