#!/usr/bin/env python3
"""
Clean inference script for RAE-based text-to-image generation.
Supports both single and multi-model comparison with elegant visualization.
"""

import argparse
import os
import time
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from scale_rae.constants import IMAGE_TOKEN_INDEX
from scale_rae.conversation import conv_templates
from scale_rae.mm_utils import tokenizer_image_token
from scale_rae.model.builder import load_pretrained_model
from scale_rae.model.multimodal_decoder import MultimodalDecoder
from scale_rae.utils import disable_torch_init


# Get repository root (assuming script is run from repo root)
REPO_ROOT = Path(__file__).parent.absolute()

# Available RAE Decoder Configurations
DECODER_CONFIGS = {
    "siglip2-so400m-web73m": {
        "pretrained_encoder_path": "google/siglip2-so400m-patch14-224",
        "general_decoder_config": str(REPO_ROOT / "decoder/XL_decoder_config.json"),
        "num_patches": 256,
        "drop_cls_token": True,
        "decoder_path": str(REPO_ROOT / "decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt"),
        "description": "SigLIP-2 So400M decoder trained on 73M web+synthetic+text data (default)",
    },
    "siglip2-so400m-web64m": {
        "pretrained_encoder_path": "google/siglip2-so400m-patch14-224",
        "general_decoder_config": str(REPO_ROOT / "decoder/XL_decoder_config.json"),
        "num_patches": 256,
        "drop_cls_token": True,
        "decoder_path": str(REPO_ROOT / "decoder/siglip2_sop14_i224_web64M_ganw3_decXL.pt"),
        "description": "SigLIP-2 So400M decoder trained on 64M web+synthetic data",
    },
    "siglip2-so400m-blip40m": {
        "pretrained_encoder_path": "google/siglip2-so400m-patch14-224",
        "general_decoder_config": str(REPO_ROOT / "decoder/XL_decoder_config.json"),
        "num_patches": 256,
        "drop_cls_token": True,
        "decoder_path": str(REPO_ROOT / "decoder/siglip2_sop14_i224_blip40M_ganw3_decXL.pt"),
        "description": "SigLIP-2 So400M decoder trained on BLIP 40M dataset",
    },
    "webssl-l": {
        "pretrained_encoder_path": "webssl-large",
        "general_decoder_config": str(REPO_ROOT / "decoder/XL_decoder_config.json"),
        "num_patches": 256,
        "drop_cls_token": True,
        "decoder_path": str(REPO_ROOT / "decoder/webssl300m_p14_i224_decXL_gan_w7.pt"),
    },
}

DEFAULT_DECODER = "siglip2-so400m-web73m"


def load_model(model_path: str, device: str = "cuda:0", decoder_name: str = DEFAULT_DECODER):
    """Load the Scale-RAE model with RAE decoder."""
    disable_torch_init()
    
    # Get decoder configuration
    if decoder_name not in DECODER_CONFIGS:
        print(f"⚠  Unknown decoder '{decoder_name}', available options:")
        for name in DECODER_CONFIGS.keys():
            print(f"   - {name}")
        print(f"Using default: {DEFAULT_DECODER}")
        decoder_name = DEFAULT_DECODER
    
    decoder_config = DECODER_CONFIGS[decoder_name]
    
    # Verify decoder files exist
    decoder_path = Path(decoder_config["decoder_path"])
    config_path = Path(decoder_config["general_decoder_config"])
    
    if not decoder_path.exists():
        raise FileNotFoundError(
            f"Decoder weights not found: {decoder_path}\n"
            f"Please ensure decoder files are in the 'decoder/' directory."
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Decoder config not found: {config_path}\n"
            f"Please ensure config file is in the 'decoder/' directory."
        )
    
    # Load model on single device to avoid device mismatch
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=Path(model_path).name,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    
    # Load RAE decoder
    decoder = MultimodalDecoder(**decoder_config)
    decoder = decoder.to(model.device)
    
    # Move normalization tensors to correct device
    if hasattr(decoder, 'image_mean') and hasattr(decoder, 'image_std'):
        decoder.image_mean = decoder.image_mean.to(model.device)
        decoder.image_std = decoder.image_std.to(model.device)
    
    print(f"✓ Model loaded: {Path(model_path).name}")
    print(f"✓ Decoder: {decoder_name}")
    return tokenizer, model, decoder


def generate_image(
    prompt: str,
    tokenizer,
    model,
    decoder,
    guidance_level: float = 1.0,
    max_new_tokens: int = 512,
):
    """Generate image from text prompt using RAE."""
    
    # Prepare conversation
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    # Get special token IDs
    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    # Generate latent features
    with torch.inference_mode():
        output_ids, image_embeds = model.generate(
            input_ids,
            images=None,
            output_image=True,
            do_sample=True,
            temperature=0.0,
            use_customize_greedy=True,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            start_image_token_id=start_image_token_id,
            end_image_token_id=end_image_token_id,
            eos_token_id=eos_token_id,
            guidance_level=guidance_level,
        )
    
    # Decode text response
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Decode image from latent features
    if image_embeds is not None and image_embeds.ndim > 1:
        image_embeds = image_embeds.unsqueeze(0)
        
        with torch.no_grad():
            # Add empty CLS token
            empty_cls_token = torch.zeros(
                (image_embeds.shape[0], 1, image_embeds.shape[-1]),
                device=image_embeds.device
            )
            image_features = torch.cat([empty_cls_token, image_embeds], dim=1)
            
            # Decode to pixels
            xs_recon = decoder(image_features)
            xs_recon = xs_recon.permute(0, 2, 3, 1).clip(0, 1).cpu().numpy()
            xs_recon = (xs_recon * 255).astype('uint8')
            image = Image.fromarray(xs_recon[0])
        
        return image, output_text
    
    return None, output_text


def add_prompt_caption(image: Image.Image, prompt: str, font_size: int = 12) -> Image.Image:
    """Add elegant prompt caption below the image."""
    img_width, img_height = image.size
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/pt-sans-fonts/PTS55F.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Wrap text
    chars_per_line = max(40, img_width // 6)
    prompt_text = f"Prompt: {prompt}"
    wrapped_lines = textwrap.wrap(prompt_text, width=chars_per_line)
    
    # Calculate dimensions
    padding = 15
    line_height = 16
    text_box_height = len(wrapped_lines) * line_height + (padding * 2)
    
    # Create new image with caption
    new_height = img_height + text_box_height + 10
    new_image = Image.new('RGB', (img_width, new_height), color='white')
    new_image.paste(image, (0, 0))
    
    # Draw text box
    draw = ImageDraw.Draw(new_image)
    text_box_y = img_height + 5
    text_box_rect = [5, text_box_y, img_width - 5, text_box_y + text_box_height]
    draw.rectangle(text_box_rect, fill='#f8f8f8', outline='#cccccc', width=1)
    
    # Draw text
    text_start_y = text_box_y + padding
    for i, line in enumerate(wrapped_lines):
        line_y = text_start_y + (i * line_height)
        draw.text((padding, line_y), line, fill='#333333', font=font)
    
    return new_image


def create_comparison(images: list, prompts: list, output_path: str):
    """Create side-by-side comparison of multiple model outputs."""
    if not images or len(images) < 2:
        return
    
    # Resize all images to same size
    max_width = max(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    resized_images = []
    for img in images:
        if img.size != (max_width, max_height):
            resized_img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Create concatenated image
    spacing = 8
    total_width = max_width * len(resized_images) + spacing * (len(resized_images) - 1)
    total_height = max_height
    
    concat_image = Image.new('RGB', (total_width, total_height), color='#f8f9fa')
    
    # Paste images
    for i, image in enumerate(resized_images):
        x_offset = i * (max_width + spacing)
        concat_image.paste(image, (x_offset, 0))
    
    concat_image.save(output_path)
    print(f"✓ Comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text using RAE-based diffusion models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (can specify multiple with comma separation)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save generated images (default: outputs)"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=DEFAULT_DECODER,
        choices=list(DECODER_CONFIGS.keys()),
        help=f"RAE decoder to use (default: {DEFAULT_DECODER})"
    )
    parser.add_argument(
        "--guidance-level",
        type=float,
        default=1.0,
        help="Guidance level for generation (default: 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle multiple models
    model_paths = [p.strip() for p in args.model_path.split(",")]
    
    print(f"\n{'='*60}")
    print(f"RAE Text-to-Image Generation")
    print(f"{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"Models: {len(model_paths)}")
    print(f"{'='*60}\n")
    
    generated_images = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] Processing: {Path(model_path).name}")
        
        # Load model
        tokenizer, model, decoder = load_model(model_path, args.device, args.decoder)
        
        # Generate image
        print("Generating image...")
        image, text_output = generate_image(
            args.prompt,
            tokenizer,
            model,
            decoder,
            guidance_level=args.guidance_level,
        )
        
        if image is not None:
            generated_images.append(image)
            
            # Save individual image with caption
            if len(model_paths) == 1:
                sample_id = f"sample_{int(time.time())}"
                captioned_image = add_prompt_caption(image, args.prompt)
                output_path = os.path.join(args.output_dir, f"{sample_id}.png")
                captioned_image.save(output_path)
                print(f"✓ Image saved: {output_path}")
            
            print(f"✓ Generated successfully")
            if text_output:
                print(f"  Text output: {text_output[:100]}...")
        else:
            print("✗ Generation failed")
        
        # Clean up
        del tokenizer, model, decoder
        torch.cuda.empty_cache()
    
    # Create comparison if multiple models
    if len(generated_images) > 1:
        sample_id = f"comparison_{int(time.time())}"
        comparison_path = os.path.join(args.output_dir, f"{sample_id}.png")
        create_comparison(generated_images, [args.prompt] * len(generated_images), comparison_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Generated {len(generated_images)} image(s)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
