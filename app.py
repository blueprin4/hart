#!/usr/bin/env python
#1.3 removed shiled module
#1.4 I've fixed the dtype mismatch issue
#1.4-1. The core error was in the addmm_ operation

import argparse
import copy
import os
import random
import uuid

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt

DESCRIPTION = (
    """# HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"""
    + """\n[\\[Paper\\]](https://arxiv.org/abs/2410.10812) [\\[Project\\]](https://hanlab.mit.edu/projects/hart) [\\[GitHub\\]](https://github.com/mit-han-lab/hart)"""
)
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_PER_PROMPT = 1


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    seed: int = 0,
    # width: int = 1024,
    # height: int = 1024,
    guidance_scale: float = 4.5,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    global text_model, text_tokenizer
    # pipe.to(device)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    prompts = [prompt]

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):

            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            infer_func = model.autoregressive_infer_cfg

            output_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )

    # bs, 3, r, r
    images = []
    sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)
        images.append(cur_img_store)

    return images, seed


@spaces.GPU(enable_queue=True)
def generate_img2img(
    prompt: str,
    input_image: Image.Image,
    condition_scale: float = 0.7,
    seed: int = 0,
    guidance_scale: float = 4.5,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    global text_model, text_tokenizer
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    prompts = [prompt]

    # Process the input image to match the model's expected input format
    # HART model expects 1024x1024 resolution images
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    
    # Handle different versions of Pillow (PIL)
    try:
        # For newer Pillow versions
        input_image = input_image.resize((1024, 1024), Image.Resampling.LANCZOS)
    except (AttributeError, ImportError):
        # For older Pillow versions
        input_image = input_image.resize((1024, 1024), Image.LANCZOS)
    
    # Convert to tensor [B, C, H, W] and normalize to [0, 1]
    # Handle potential errors in numpy conversion
    try:
        input_np = np.array(input_image)
        if input_np.ndim != 3 or input_np.shape[2] != 3:
            print(f"Warning: Unexpected input image shape: {input_np.shape}, converting to RGB")
            input_image = input_image.convert('RGB')
            input_np = np.array(input_image)
        
        # Based on the error, let's specifically use float32 here
        # The model's VAE encoder requires inputs to be in float32 format for matrix math
        input_tensor = (
            torch.from_numpy(input_np)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)  # Explicitly use float32
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
    except Exception as e:
        print(f"Error processing input image: {e}")
        # Create a fallback red image if conversion fails
        # Ensure we're using float32 for the fallback image too
        input_tensor = torch.zeros(1, 3, 1024, 1024, device=device, dtype=torch.float32)
        input_tensor[0, 0, :, :] = 1.0  # Red channel

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):
            # Important: Do NOT convert input_tensor to float16 here!
            # The VAE encoder in the model expects float32 input
            # The conversion to appropriate dtype is handled inside the model

            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            infer_func = model.autoregressive_infer_cfg

            output_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
                input_image=input_tensor,
                condition_scale=condition_scale,
            )

    # bs, 3, r, r
    images = []
    sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)
        images.append(cur_img_store)

    return images, seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    model = HARTForT2I(config)
    model.eval()
    weights_path = "ema_model.bin" if args.use_ema else "pytorch_model.bin"
    model.load_state_dict(torch.load(os.path.join(args.model_path, weights_path), map_location="cpu"))
    model = model.to(device)
    model.eval()

    if args.use_ema:
        model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(
        args.text_model_path, torch_dtype=torch.float16
    ).to(device)
    text_model.eval()
    text_tokenizer_max_length = args.max_token_length

    examples = [
        "melting apple",
        "neon holography crystal cat",
        "A dog that has been meditating all the time",
        "An astronaut riding a horse on the moon, oil painting by Van Gogh.",
        "8k uhd A man looks up at the starry sky, lonely and ethereal, Minimalism, Chaotic composition Op Art",
        "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.",
        "Steampunk makeup, in the style of vray tracing, colorful impasto, uhd image, indonesian art, fine feather details with bright red and yellow and green and pink and orange colours, intricate patterns and details, dark cyan and amber makeup. Rich colourful plumes. Victorian style.",
    ]

    css = """
    .gradio-container{max-width: 560px !important}
    h1{text-align:center}
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown(DESCRIPTION)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
        )
        
        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                with gr.Group():
                    with gr.Row():
                        t2i_prompt = gr.Text(
                            label="Prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter your prompt",
                            container=False,
                        )
                        t2i_run_button = gr.Button("Run", scale=0)

                    t2i_result = gr.Gallery(
                        label="Result",
                        columns=NUM_IMAGES_PER_PROMPT,
                        show_label=False,
                    )
                    with gr.Accordion("Advanced options", open=False):
                        t2i_seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=args.seed,
                        )
                        t2i_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        with gr.Row():
                            t2i_guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.1,
                                maximum=20,
                                step=0.1,
                                value=4.5,
                            )

                gr.Examples(
                    examples=examples,
                    inputs=t2i_prompt,
                    outputs=[t2i_result, t2i_seed],
                    fn=generate,
                    cache_examples=CACHE_EXAMPLES,
                )

                gr.on(
                    triggers=[
                        t2i_prompt.submit,
                        t2i_run_button.click,
                    ],
                    fn=generate,
                    inputs=[
                        t2i_prompt,
                        t2i_seed,
                        t2i_guidance_scale,
                        t2i_randomize_seed,
                    ],
                    outputs=[t2i_result, t2i_seed],
                    api_name="run_t2i",
                )
            
            with gr.TabItem("Image to Image"):
                with gr.Group():
                    with gr.Row():
                        i2i_prompt = gr.Text(
                            label="Prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Describe the changes you want to make to the image",
                            container=False,
                        )
                        i2i_run_button = gr.Button("Run", scale=0)
                    
                    i2i_input_image = gr.Image(
                        label="Input Image", 
                        type="pil", 
                        height=384
                    )
                    
                    i2i_result = gr.Gallery(
                        label="Result",
                        columns=NUM_IMAGES_PER_PROMPT,
                        show_label=False,
                    )
                    
                    with gr.Accordion("Advanced options", open=False):
                        i2i_condition_scale = gr.Slider(
                            label="Conditioning Scale",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.7,
                            info="How much to preserve from the original image (higher = more faithful to original)"
                        )
                        i2i_seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=args.seed,
                        )
                        i2i_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        with gr.Row():
                            i2i_guidance_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.1,
                                maximum=20,
                                step=0.1,
                                value=4.5,
                            )
                
                gr.on(
                    triggers=[
                        i2i_prompt.submit,
                        i2i_run_button.click,
                    ],
                    fn=generate_img2img,
                    inputs=[
                        i2i_prompt,
                        i2i_input_image,
                        i2i_condition_scale,
                        i2i_seed,
                        i2i_guidance_scale,
                        i2i_randomize_seed,
                    ],
                    outputs=[i2i_result, i2i_seed],
                    api_name="run_i2i",
                )

    demo.queue(max_size=20).launch(share=True)