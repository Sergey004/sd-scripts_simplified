import gradio as gr
import subprocess
import shlex
import re

# Function to run training

def run_training(
    project_name,
    base_dir,
    run_steps,
    base_model,
    base_vae,
    tagging_method,
    tagger_threshold,
    tagger_batch_size,
    tagger_blacklist,
    caption_extension,
    activation_tag,
    remove_tags,
    remove_duplicate_tags,
    resolution,
    keep_tokens,
    preferred_unit,
    how_many,
    save_every_n_epochs,
    keep_only_last_n_epochs,
    unet_lr,
    text_encoder_lr,
    lr_scheduler,
    lr_scheduler_num_cycles,
    lr_warmup_ratio,
    min_snr_gamma,
    multinoise,
    lora_type,
    network_dim,
    network_alpha,
    auto_vram_params,
    auto_repeats,
    cross_attention,
    cache_latents,
    cache_latents_to_disk,
    gradient_checkpointing,
    use_recommended_optimizer_args,
    seed,
    lowram,
    max_bucket_reso,
    num_cpu_threads
):
    cmd = f"python master_train.py " \
          f"--project_name '{project_name}' " \
          f"--base_dir '{base_dir}' " \
          f"--run_steps '{run_steps}' " \
          f"--base_model '{base_model}' " \
          f"--base_vae '{base_vae}' " \
          f"--tagging_method '{tagging_method}' " \
          f"--tagger_threshold {tagger_threshold} " \
          f"--tagger_batch_size {tagger_batch_size} " \
          f"--tagger_blacklist '{tagger_blacklist}' " \
          f"--caption_extension '{caption_extension}' " \
          f"--activation_tag '{activation_tag}' " \
          f"--remove_tags '{remove_tags}' " \
          f"--remove_duplicate_tags " if remove_duplicate_tags else "" \
          f"--resolution {resolution} " \
          f"--keep_tokens {keep_tokens} " \
          f"--preferred_unit '{preferred_unit}' " \
          f"--how_many {how_many} " \
          f"--save_every_n_epochs {save_every_n_epochs} " \
          f"--keep_only_last_n_epochs {keep_only_last_n_epochs} " \
          f"--unet_lr {unet_lr} " \
          f"--text_encoder_lr {text_encoder_lr} " \
          f"--lr_scheduler '{lr_scheduler}' " \
          f"--lr_scheduler_num_cycles {lr_scheduler_num_cycles} " \
          f"--lr_warmup_ratio {lr_warmup_ratio} " \
          f"--min_snr_gamma {min_snr_gamma} " \
          f"--multinoise " if multinoise else "" \
          f"--lora_type '{lora_type}' " \
          f"--network_dim {network_dim} " \
          f"--network_alpha {network_alpha} " \
          f"--auto_vram_params " if auto_vram_params else "" \
          f"--auto_repeats " if auto_repeats else "" \
          f"--cross_attention '{cross_attention}' " \
          f"--cache_latents " if cache_latents else "" \
          f"--cache_latents_to_disk " if cache_latents_to_disk else "" \
          f"--gradient_checkpointing " if gradient_checkpointing else "" \
          f"--use_recommended_optimizer_args " if use_recommended_optimizer_args else "" \
          f"--seed {seed} " \
          f"--lowram " if lowram else "" \
          f"--max_bucket_reso {max_bucket_reso} " \
          f"--num_cpu_threads {num_cpu_threads} "
    # Remove double spaces
    cmd = ' '.join(cmd.split())
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def decode_sh_file(sh_text):
    """
    Parse .sh file and extract arguments for master_train.py
    Returns a dict with parameter names and values
    """
    # Find the line with python master_train.py ...
    match = re.search(r'python +master_train.py +(.+)', sh_text, re.DOTALL)
    if not match:
        return {"error": "No master_train.py command found in script."}
    args_line = match.group(1)
    # Remove line continuations and backslashes
    args_line = args_line.replace('\\', ' ')
    # Split by --param_name
    args = re.findall(r'--([\w_\-]+)(?: +(["\"][^"\"]*["\"]|\S+))?', args_line)
    params = {}
    for key, value in args:
        value = value.strip('"\'') if value else True
        params[key] = value
    return params

def fill_fields_from_sh(sh_text):
    params = decode_sh_file(sh_text)
    if "error" in params:
        return [gr.update() for _ in range(41)] + [params["error"]]
    # Map params to UI fields (order must match UI inputs)
    field_order = [
        "project_name", "base_dir", "run_steps", "base_model", "base_vae", "tagging_method", "tagger_threshold", "tagger_batch_size", "tagger_blacklist", "caption_extension", "activation_tag", "remove_tags", "remove_duplicate_tags", "resolution", "keep_tokens", "preferred_unit", "how_many", "save_every_n_epochs", "keep_only_last_n_epochs", "unet_lr", "text_encoder_lr", "lr_scheduler", "lr_scheduler_num_cycles", "lr_warmup_ratio", "min_snr_gamma", "multinoise", "lora_type", "network_dim", "network_alpha", "auto_vram_params", "auto_repeats", "cross_attention", "cache_latents", "cache_latents_to_disk", "gradient_checkpointing", "use_recommended_optimizer_args", "seed", "lowram", "max_bucket_reso", "num_cpu_threads"
    ]
    result = []
    for field in field_order:
        val = params.get(field, None)
        if val is None:
            result.append(gr.update())
        elif val is True:
            result.append(True)
        else:
            # Try to convert to float/int if possible
            try:
                if '.' in val:
                    result.append(float(val))
                else:
                    result.append(int(val))
            except Exception:
                result.append(val)
    result.append("Decoded successfully.")
    return result

with gr.Blocks() as demo:
    gr.Markdown("# Gradio interface for master_train.py")
    with gr.Tab("Training UI"):
        with gr.Row():
            project_name = gr.Textbox(label="Project Name", value="Lora_name")
            base_dir = gr.Textbox(label="Base Directory", value="./Loras")
            run_steps = gr.Textbox(label="Steps (tag,curate,config,train)", value="tag,curate,config,train")
        with gr.Row():
            base_model = gr.Textbox(label="Base Model (path or URL)", value="https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors", info="You can use a local file path or a URL")
            base_vae = gr.Textbox(label="Base VAE", value="stabilityai/sdxl-vae")
        with gr.Row():
            tagging_method = gr.Textbox(label="Tagging Method", value="wd14")
            tagger_threshold = gr.Number(label="Tagger Threshold", value=0.35)
            tagger_batch_size = gr.Number(label="Tagger Batch Size", value=8)
        tagger_blacklist = gr.Textbox(label="Tagger Blacklist", value="bangs, breasts, multicolored hair, two-tone hair, gradient hair, virtual youtuber, parody, style parody, official alternate costume, official alternate hairstyle, official alternate hair length, alternate costume, alternate hairstyle, alternate hair length, alternate hair color")
        caption_extension = gr.Textbox(label="Caption Extension", value=".txt")
        activation_tag = gr.Textbox(label="Activation Tag", value="put_yours")
        remove_tags = gr.Textbox(label="Remove Tags", value="candy, musical note, gradient, white background, background, green eyes, heart, gradient background, solo, artist name, traditional media, multicolored background, checkered background, purple background, looking at viewer, simple background, male focus, brown eyes, feet out of frame, underwear only, window, sitting, couch, night sky, night, starry sky, brown fur")
        remove_duplicate_tags = gr.Checkbox(label="Remove Duplicate Tags", value=True)
        resolution = gr.Number(label="Resolution", value=1024)
        keep_tokens = gr.Number(label="Keep Tokens", value=1)
        preferred_unit = gr.Textbox(label="Preferred Unit", value="Epochs")
        how_many = gr.Number(label="How Many", value=10)
        save_every_n_epochs = gr.Number(label="Save Every N Epochs", value=1)
        keep_only_last_n_epochs = gr.Number(label="Keep Only Last N Epochs", value=10)
        unet_lr = gr.Textbox(label="UNet LR", value="3e-4")
        text_encoder_lr = gr.Textbox(label="Text Encoder LR", value="6e-5")
        lr_scheduler = gr.Textbox(label="LR Scheduler", value="cosine_with_restarts")
        lr_scheduler_num_cycles = gr.Number(label="LR Scheduler Num Cycles", value=3)
        lr_warmup_ratio = gr.Number(label="LR Warmup Ratio", value=0.05)
        min_snr_gamma = gr.Number(label="Min SNR Gamma", value=8.0)
        multinoise = gr.Checkbox(label="Multinoise", value=True)
        lora_type = gr.Textbox(label="LoRA Type", value="LoRA")
        network_dim = gr.Number(label="Network Dim", value=32)
        network_alpha = gr.Number(label="Network Alpha", value=16)
        auto_vram_params = gr.Checkbox(label="Auto VRAM Params", value=True)
        auto_repeats = gr.Checkbox(label="Auto Repeats", value=True)
        cross_attention = gr.Textbox(label="Cross Attention", value="sdpa")
        cache_latents = gr.Checkbox(label="Cache Latents", value=True)
        cache_latents_to_disk = gr.Checkbox(label="Cache Latents To Disk", value=True)
        gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True)
        use_recommended_optimizer_args = gr.Checkbox(label="Use Recommended Optimizer Args", value=True)
        seed = gr.Number(label="Seed", value=42)
        lowram = gr.Checkbox(label="Low RAM", value=True)
        max_bucket_reso = gr.Number(label="Max Bucket Reso", value=4096)
        num_cpu_threads = gr.Number(label="Num CPU Threads", value=12)
        
        output = gr.Textbox(label="Execution Log", lines=20)
        
        btn = gr.Button("Start Training")
        
        btn.click(
            run_training,
            inputs=[project_name, base_dir, run_steps, base_model, base_vae, tagging_method, tagger_threshold, tagger_batch_size, tagger_blacklist, caption_extension, activation_tag, remove_tags, remove_duplicate_tags, resolution, keep_tokens, preferred_unit, how_many, save_every_n_epochs, keep_only_last_n_epochs, unet_lr, text_encoder_lr, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_ratio, min_snr_gamma, multinoise, lora_type, network_dim, network_alpha, auto_vram_params, auto_repeats, cross_attention, cache_latents, cache_latents_to_disk, gradient_checkpointing, use_recommended_optimizer_args, seed, lowram, max_bucket_reso, num_cpu_threads],
            outputs=output
        )

    with gr.Tab("SH Decoder"):
        gr.Markdown("## Paste your .sh launch script below. The UI fields will be filled automatically.")
        sh_input = gr.Textbox(label="Paste .sh file content here", lines=20)
        decode_btn = gr.Button("Decode .sh file and fill UI fields")
        decode_status = gr.Textbox(label="Decoder Status", interactive=False)
        decode_btn.click(
            fill_fields_from_sh,
            inputs=[sh_input],
            outputs=[project_name, base_dir, run_steps, base_model, base_vae, tagging_method, tagger_threshold, tagger_batch_size, tagger_blacklist, caption_extension, activation_tag, remove_tags, remove_duplicate_tags, resolution, keep_tokens, preferred_unit, how_many, save_every_n_epochs, keep_only_last_n_epochs, unet_lr, text_encoder_lr, lr_scheduler, lr_scheduler_num_cycles, lr_warmup_ratio, min_snr_gamma, multinoise, lora_type, network_dim, network_alpha, auto_vram_params, auto_repeats, cross_attention, cache_latents, cache_latents_to_disk, gradient_checkpointing, use_recommended_optimizer_args, seed, lowram, max_bucket_reso, num_cpu_threads, decode_status]
        )

demo.launch()
