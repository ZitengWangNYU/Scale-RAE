#!/usr/bin/env python
import argparse
import os
from re import I
import sys
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download

# Allow loading of large images (disable decompression bomb check)
Image.MAX_IMAGE_PIXELS = None

from scale_rae.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from scale_rae.conversation import conv_templates
from scale_rae.mm_utils import tokenizer_image_token
from scale_rae.model.multimodal_decoder import MultimodalDecoder
from utils.load_model import load_scale_rae_model


# Default Hugging Face repository for decoder
DEFAULT_DECODER_REPO = "nyu-visionx/siglip2_decoder"
DEFAULT_MODEL_PATH = "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B"


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def make_sample_id(prefix: str = "sample") -> str:
    import time
    return f"{prefix}_{int(time.time())}"


def save_images(images: List[Image.Image], output_dir: str, sample_id: str) -> List[str]:
    paths: List[str] = []
    if not images:
        return paths
    if len(images) == 1:
        out_path = os.path.join(output_dir, f"{sample_id}.png")
        images[0].save(out_path)
        return [out_path]
    for i, img in enumerate(images):
        out_path = os.path.join(output_dir, f"{sample_id}_{i:02d}.png")
        img.save(out_path)
        paths.append(out_path)
    return paths


def write_manifest(manifest_path: str, manifest: Dict) -> None:
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_model(model_path: str):
    tokenizer, model, image_processor, context_len = load_scale_rae_model(model_path)
    return tokenizer, model, image_processor, context_len


def prepare_special_token_ids(tokenizer) -> Tuple[int, int, int]:
    start_image_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    end_image_token_id = tokenizer.convert_tokens_to_ids("<im_end>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return start_image_token_id, end_image_token_id, eos_token_id


def _maybe_add_image_tokens_to_prompt(qs: str, num_frames: int, use_im_se: bool) -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if use_im_se:
            qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se * num_frames)
        else:
            qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN * num_frames)
    else:
        if use_im_se:
            qs = image_token_se * num_frames + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * num_frames + "\n" + qs
    return qs


def build_prompt(prompt_text: str, model_config, with_image: bool, num_frames: int = 1) -> str:
    qs = prompt_text
    if with_image:
        qs = _maybe_add_image_tokens_to_prompt(qs, num_frames=num_frames, use_im_se=bool(model_config.mm_use_im_start_end))
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def tokenize_prompt(prompt: str, tokenizer, device: torch.device) -> torch.Tensor:
    """Tokenize prompt and move to specified device."""
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    input_ids = input_ids.to(device)
    return input_ids


def load_image_rgb(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def preprocess_single_image(
    image: Image.Image,
    image_processor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image_tensor = image_processor[0].preprocess(image, return_tensors="pt")["pixel_values"][0]
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device, dtype=dtype)
    return image_tensor, image.size


def detect_vae_mode(model, model_path: Optional[str] = None) -> bool:
    cfg_flag = bool(getattr(model.config, "generation_alignment_tower", None))
    if cfg_flag:
        return True
    if model_path:
        if "vae" in str(model_path).lower():
            return True
    return False


def _parse_flux_repo(name: str) -> str:
    base = name
    if "-res" in base:
        base = base.split("-res")[0]
    if "-interp" in base:
        base = base.split("-interp")[0]
    return base


def build_decoder(model, model_path: Optional[str] = None, decoder_repo_id: str = DEFAULT_DECODER_REPO):
    device = model.device
    if detect_vae_mode(model, model_path=model_path):
        vae_repo_name = model.config.generation_alignment_tower
        vae_base_repo = _parse_flux_repo(vae_repo_name)
        vae = AutoencoderKL.from_pretrained(vae_base_repo, subfolder="vae", device_map=None, low_cpu_mem_usage=False)
        vae = vae.to(device)
        vae.eval()
        try:
            if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
                from torch import nn as _nn
                vae.quant_conv = _nn.Identity()
            if hasattr(vae, "post_quant_conv") and vae.post_quant_conv is not None:
                from torch import nn as _nn
                vae.post_quant_conv = _nn.Identity()
        except Exception:
            pass
        return {
            "mode": "vae",
            "vae": vae,
            "tokens_per_image": int(getattr(model, "num_image_tokens", 256)),
            "patch_size": int(getattr(model.config, "patch_size", 2)),
        }

    # RAE Mode - Download decoder from Hugging Face
    print(f"Loading decoder from Hugging Face: {decoder_repo_id}...")
    
    # Download/Cache files from HF Hub
    config_path = hf_hub_download(repo_id=decoder_repo_id, filename="config.json")
    ckpt_path = hf_hub_download(repo_id=decoder_repo_id, filename="model.pt")
    
    # Get encoder path from model config (strip interpolation suffix)
    encoder_path = model.config.mm_vision_tower_aux_list[0].split('-interp')[0]
    
    decoder_params = {
        "pretrained_encoder_path": encoder_path,
        "general_decoder_config": config_path,
        "num_patches": 256,
        "drop_cls_token": True,
        "decoder_path": ckpt_path,
    }
    
    decoder = MultimodalDecoder(**decoder_params)
    decoder = decoder.to(device)
    if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
        decoder.image_mean = decoder.image_mean.to(device)
        decoder.image_std = decoder.image_std.to(device)
    return {"mode": "siglip", "decoder": decoder}


def decode_image_embeds(model, image_embeds: torch.Tensor, bundle: Dict[str, Any]) -> List[Image.Image]:
    if image_embeds is None or image_embeds.ndim == 1:
        return []

    if bundle["mode"] == "vae":
        vae: AutoencoderKL = bundle["vae"]
        tokens_per_image = int(bundle["tokens_per_image"]) if bundle.get("tokens_per_image") is not None else 256
        patch_size = int(bundle["patch_size"]) if bundle.get("patch_size") is not None else 2
        T, D = image_embeds.shape
        if T % tokens_per_image != 0:
            return []
        B_img = T // tokens_per_image
        tokens = image_embeds.view(B_img, tokens_per_image, D).to(model.device)
        p = patch_size
        H = W = int((tokens_per_image * (p * p)) ** 0.5)
        assert tokens_per_image * (p * p) == H * W, "Incompatible dimensions for folding"
        latents = tokens.transpose(1, 2)
        latents = F.fold(latents, output_size=(H, W), kernel_size=p, stride=p)
        latents = latents.to(vae.dtype)
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        with torch.no_grad():
            recon = vae.decode(latents, return_dict=False)[0]
            recon = (recon + 1.0) / 2.0
        images: List[Image.Image] = []
        for i in range(recon.shape[0]):
            arr = (recon[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            images.append(Image.fromarray(arr))
        return images

    decoder: MultimodalDecoder = bundle["decoder"]
    image_embeds_batched = image_embeds.unsqueeze(0)
    with torch.no_grad():
        empty_cls_token = torch.zeros((image_embeds_batched.shape[0], 1, image_embeds_batched.shape[-1]), device=image_embeds_batched.device)
        image_features = torch.cat([empty_cls_token, image_embeds_batched], dim=1)
        xs_recon = decoder(image_features)
        xs_recon = xs_recon.permute(0, 2, 3, 1).clip(0, 1).cpu().numpy()
        xs_recon = (xs_recon * 255).astype("uint8")
        images = [Image.fromarray(x) for x in xs_recon]
    return images


def _common_gen_kwargs(
    start_image_token_id: int,
    end_image_token_id: int,
    eos_token_id: int,
    guidance_level: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    return dict(
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


def cmd_t2i(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    tokenizer, model, image_processor, context_len = load_model(args.model_path)
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)

    prompt_text = args.prompt.replace(IMAGE_PLACEHOLDER, "")
    prompt = build_prompt(prompt_text, model_config=model.config, with_image=False)
    input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)

    with torch.inference_mode():
        output_ids, image_embeds = model.generate(
            input_ids,
            images=None,
            **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens),
        )

    # Decode text output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    sample_id = make_sample_id("sample")

    latent_path = None
    if args.save_latent and image_embeds is not None:
        latent_path = os.path.join(args.output_dir, f"{sample_id}_latent.pt")
        torch.save(image_embeds.detach().to("cpu"), latent_path)

    image_paths: List[str] = []
    if not args.skip_decoder and image_embeds is not None:
        bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)
        images = decode_image_embeds(model, image_embeds, bundle)
        image_paths = save_images(images, args.output_dir, sample_id)

    manifest: Dict[str, Any] = dict(
        mode="t2i",
        model_path=args.model_path,
        prompt=args.prompt,
        guidance_level=args.guidance_level,
        skip_decoder=bool(args.skip_decoder),
        output_images=image_paths,
        latent_path=latent_path,
        output_text=output_text,
    )

    post_qa_prompt = ""
    if args.post_qa_template_path:
        with open(args.post_qa_template_path, "r") as f:
            post_qa_prompt = f.read()
    elif args.post_qa_prompt:
        post_qa_prompt = args.post_qa_prompt

    # Optional post-step: QA using latent or decoded image
    if post_qa_prompt:
        output_text = None
        image_embeds = image_embeds.unsqueeze(0)

        if args.post_qa_mode == "latent":
            prompt_post = build_prompt(post_qa_prompt, model_config=model.config, with_image=True, num_frames=1)
            input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)
            with torch.inference_mode():
                output_ids_post, _ = model.generate(
                    input_ids_post,
                    image_embeds=image_embeds,
                    **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                )
            output_text = tokenizer.decode(output_ids_post[0], skip_special_tokens=True)
        elif args.post_qa_mode == "image":
            if not image_paths and (image_embeds is not None) and (not args.skip_decoder):
                bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)
                images_now = decode_image_embeds(model, image_embeds, bundle)
                image_paths = save_images(images_now, args.output_dir, sample_id)
            if image_paths:
                img_path = image_paths[0]
                image = load_image_rgb(img_path)
                images_tensor, size = preprocess_single_image(image, image_processor, device=model.device, dtype=model.dtype)
                # print("images_tensor shape is", images_tensor.shape, "dtype is", images_tensor.dtype)
                prompt_post = build_prompt(post_qa_prompt, model_config=model.config, with_image=True, num_frames=1)
                input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)
                with torch.inference_mode():
                    output_ids_post, _ = model.generate(
                        input_ids_post,
                        images=images_tensor,
                        **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                    )
                output_text = tokenizer.decode(output_ids_post[0], skip_special_tokens=True)
        else:
            raise SystemExit(f"Unknown post-qa mode: {args.post_qa_mode}")

        if output_text is not None:
            manifest["post_qa"] = dict(
                mode=args.post_qa_mode,
                prompt=post_qa_prompt,
                output_text=output_text,
            )
            print(output_text)

    manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{sample_id}_manifest.json"))
    write_manifest(manifest_path, manifest)

    # Final summary prints with absolute paths
    if image_paths:
        image_paths_abs = [os.path.abspath(p) for p in image_paths]
        print("Images saved to: " + ", ".join(image_paths_abs))
    if latent_path:
        print("Latent saved to: " + os.path.abspath(latent_path))
    print("Manifest saved to: " + manifest_path)
    print("\nText output:\n" + output_text)


def cmd_latent(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    tokenizer, model, image_processor, context_len = load_model(args.model_path)
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    try:
        image_embeds = torch.load(args.latent, map_location="cpu")
    except Exception:
        raise SystemExit(f"Failed to load latent from: {args.latent}")
    if not isinstance(image_embeds, torch.Tensor):
        raise SystemExit(f"Latent file is not a tensor: {args.latent}")
    image_embeds = image_embeds.to(device=model.device, dtype=model.dtype)

    sample_id = make_sample_id("sample")

    if args.action == "decode":
        if args.skip_decoder:
            manifest = dict(
                mode="latent-decode",
                model_path=args.model_path,
                latent_path=args.latent,
                skip_decoder=True,
                output_images=[],
            )
            manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{sample_id}_manifest.json"))
            write_manifest(manifest_path, manifest)
            # Final summary prints
            print("Latent used: " + os.path.abspath(args.latent))
            print("Manifest saved to: " + manifest_path)
            return
        bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)
        images = decode_image_embeds(model, image_embeds, bundle)
        image_paths = save_images(images, args.output_dir, sample_id)
        manifest = dict(
            mode="latent-decode",
            model_path=args.model_path,
            latent_path=args.latent,
            skip_decoder=False,
            output_images=image_paths,
        )
        manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{sample_id}_manifest.json"))
        write_manifest(manifest_path, manifest)
        # Final summary prints
        if image_paths:
            image_paths_abs = [os.path.abspath(p) for p in image_paths]
            print("Images saved to: " + ", ".join(image_paths_abs))
        print("Latent used: " + os.path.abspath(args.latent))
        print("Manifest saved to: " + manifest_path)
        return

    if args.action in ("qa", "continue"):
        if not args.prompt:
            raise SystemExit("--prompt is required for action qa/continue")
        prompt = build_prompt(args.prompt, model_config=model.config, with_image=True, num_frames=1)
        input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)
        # Add batch dimension if needed
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(0)
        with torch.inference_mode():
            output_ids, _ = model.generate(
                input_ids,
                image_embeds=image_embeds,
                **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens_qa),
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        manifest = dict(
            mode=f"latent-{args.action}",
            model_path=args.model_path,
            latent_path=args.latent,
            prompt=args.prompt,
            guidance_level=args.guidance_level,
            output_text=output_text,
        )
        manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{sample_id}_manifest.json"))
        write_manifest(manifest_path, manifest)
        # Final summary prints
        print("Latent used: " + os.path.abspath(args.latent))
        print("Manifest saved to: " + manifest_path)
        print(output_text)
        return

    raise SystemExit(f"Unknown latent action: {args.action}")


def cmd_img(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_dir(args.output_dir)

    tokenizer, model, image_processor, context_len = load_model(args.model_path)
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)

    image = load_image_rgb(args.image)
    images_tensor, size = preprocess_single_image(image, image_processor, device=model.device, dtype=model.dtype)

    prompt = build_prompt(args.prompt, model_config=model.config, with_image=True, num_frames=1)
    input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)

    with torch.inference_mode():
        output_ids, image_embeds = model.generate(
            input_ids,
            images=images_tensor,
            **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens),
        )

    # Decode text output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    sample_id = make_sample_id("sample")
    latent_path = None
    if args.save_latent and image_embeds is not None:
        latent_path = os.path.join(args.output_dir, f"{sample_id}_latent.pt")
        torch.save(image_embeds.detach().to("cpu"), latent_path)

    image_paths: List[str] = []
    if not args.skip_decoder and image_embeds is not None:
        bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=args.decoder_repo)
        images = decode_image_embeds(model, image_embeds, bundle)
        image_paths = save_images(images, args.output_dir, sample_id)

    manifest: Dict[str, Any] = dict(
        mode="img",
        model_path=args.model_path,
        input_image=args.image,
        prompt=args.prompt,
        guidance_level=args.guidance_level,
        skip_decoder=bool(args.skip_decoder),
        output_images=image_paths,
        latent_path=latent_path,
        output_text=output_text,
    )
    manifest_path = os.path.abspath(os.path.join(args.output_dir, f"{sample_id}_manifest.json"))
    write_manifest(manifest_path, manifest)
    # Final summary prints
    if image_paths:
        image_paths_abs = [os.path.abspath(p) for p in image_paths]
        print("Images saved to: " + ", ".join(image_paths_abs))
    if latent_path:
        print("Latent saved to: " + os.path.abspath(latent_path))
    print("Manifest saved to: " + manifest_path)
    print("\nText output:\n" + output_text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scale-RAE inference CLI (t2i, latent, img)")
    sp = p.add_subparsers(dest="cmd", required=True)

    # t2i
    p_t2i = sp.add_parser("t2i", help="Text-to-image generation")
    p_t2i.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_t2i.add_argument("--prompt", required=True)
    p_t2i.add_argument("--output-dir", default="outputs")
    p_t2i.add_argument("--guidance-level", type=float, default=1.0)
    p_t2i.add_argument("--skip-decoder", action="store_true")
    p_t2i.add_argument("--save-latent", action="store_true")
    p_t2i.add_argument("--max-new-tokens", type=int, default=512)
    p_t2i.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")
    p_t2i.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    # Optional post-step (QA using latent or decoded image)
    p_t2i.add_argument("--post-qa-template-path", default=None)
    p_t2i.add_argument("--post-qa-prompt", default=None)
    p_t2i.add_argument("--post-qa-mode", choices=["latent", "image"], default="latent")
    p_t2i.add_argument("--post-qa-max-new-tokens", type=int, default=64)

    # latent
    p_lat = sp.add_parser("latent", help="Operate on saved latents (decode, qa, continue)")
    p_lat.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_lat.add_argument("--latent", required=True, help="Path to saved latent .pt")
    p_lat.add_argument("--action", required=True, choices=["decode", "qa", "continue"])
    p_lat.add_argument("--prompt", default=None, help="Prompt for qa/continue")
    p_lat.add_argument("--output-dir", default="outputs")
    p_lat.add_argument("--guidance-level", type=float, default=1.0)
    p_lat.add_argument("--skip-decoder", action="store_true")
    p_lat.add_argument("--max-new-tokens-qa", type=int, default=64)
    p_lat.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")
    p_lat.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # img
    p_img = sp.add_parser("img", help="Use an input image with a prompt")
    p_img.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_img.add_argument("--image", required=True)
    p_img.add_argument("--prompt", required=True)
    p_img.add_argument("--output-dir", default="outputs")
    p_img.add_argument("--guidance-level", type=float, default=1.0)
    p_img.add_argument("--skip-decoder", action="store_true")
    p_img.add_argument("--save-latent", action="store_true")
    p_img.add_argument("--max-new-tokens", type=int, default=512)
    p_img.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")
    p_img.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "t2i":
        return cmd_t2i(args)
    if args.cmd == "latent":
        return cmd_latent(args)
    if args.cmd == "img":
        return cmd_img(args)

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()


