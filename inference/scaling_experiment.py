#!/usr/bin/env python
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

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
from utils.selectors import ImageRewardSelector


# Default Hugging Face repository for decoder
DEFAULT_DECODER_REPO = "nyu-visionx/siglip2_decoder"
DEFAULT_MODEL_PATH = "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B"


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def make_sample_id(prefix: str = "sample") -> str:
    import time
    return f"{prefix}_{int(time.time())}"


def save_images(images: List[Image.Image], sample_dir: str, start_count: int = 0) -> List[str]:
    """Save images using geneval_eval_ddp.py style naming.
    
    Args:
        images: List of PIL images to save
        sample_dir: Directory to save images in
        start_count: Starting index for image numbering
    
    Returns:
        List of saved image paths
    """
    paths: List[str] = []
    if not images:
        return paths
    
    os.makedirs(sample_dir, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(sample_dir, f"{start_count + i:05}.png")
        img.save(out_path)
        paths.append(out_path)
    return paths


def write_metadata(metadata_path: str, metadata: Dict) -> None:
    """Write metadata in jsonl format like geneval_eval_ddp.py"""
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def load_model(model_path: str, device: str = "cuda"):
    """Load model. Uses multi-GPU automatically if available."""
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


def build_prompt(prompt_text: str, model_config, with_image: bool, gen_image: bool = False, num_frames: int = 1) -> str:
    qs = prompt_text
    if with_image:
        qs = _maybe_add_image_tokens_to_prompt(qs, num_frames=num_frames, use_im_se=bool(model_config.mm_use_im_start_end))
    if gen_image:
        qs = "Generate an image of " + qs
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
    ).unsqueeze(0).to(device)
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
    # print(f"Loading decoder from Hugging Face: {decoder_repo_id}...")
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


def get_score(output_text, output_logits, output_scores, output_ids, tokenizer):
    """Extract yes/no confidence from model output scores.
    
    Args:
        output_text: The decoded text output from the model
        output_score: Tuple of scalar logit tensors (one per generated token)
                     Each tensor is shape (1,) containing the logit of the selected token
        output_ids: The generated token IDs
        tokenizer: The tokenizer to convert tokens to IDs
    
    Returns:
        dict with:
            - 'answer': 'yes', 'no', or 'uncertain'
            - 'yes_logit': logit value if yes token was generated, else None
            - 'no_logit': logit value if no token was generated, else None
            - 'max_logit': maximum logit value across all tokens
            - 'avg_logit': average logit value across all tokens
            - 'text': the output text
    """
    
    # Validate input
    assert len(output_scores) == len(output_ids), "Number of scores and tokens must match"
    
    # Get token IDs for yes/no (try different capitalizations and with/without spaces)
    yes_tokens = set()
    no_tokens = set()
    
    # Try different variations of "yes"
    for variant in ['yes', 'Yes', 'YES', ' yes', ' Yes', ' YES']:
        try:
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(variant))
            if token_ids:
                if isinstance(token_ids, list):
                    yes_tokens.update(token_ids)
                else:
                    yes_tokens.add(token_ids)
        except:
            pass
    
    # Try different variations of "no"
    for variant in ['no', 'No', 'NO', ' no', ' No', ' NO']:
        try:
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(variant))
            if token_ids:
                if isinstance(token_ids, list):
                    no_tokens.update(token_ids)
                else:
                    no_tokens.add(token_ids)
        except:
            pass

    # Extract generated token IDs
    if hasattr(output_ids, 'tolist'):
        generated_token_ids = output_ids.tolist() if output_ids.dim() == 1 else output_ids[0].tolist()
    else:
        generated_token_ids = output_ids
    
    # Find yes/no tokens in generated sequence and their logits
    yes_logit = None
    no_logit = None
    yes_conf_score = None
    no_conf_score = None
    all_logits = []
    all_conf_scores = []
    
    for i, (token_id, logit_tensor, score_tensor) in enumerate(zip(generated_token_ids, output_logits, output_scores)):
        logit_value = logit_tensor.item() if hasattr(logit_tensor, 'item') else float(logit_tensor[0])
        all_logits.append(logit_value)
        conf_score_value = score_tensor.item() if hasattr(score_tensor, 'item') else float(score_tensor[0])
        all_conf_scores.append(conf_score_value)
        if token_id in yes_tokens:
            yes_logit = logit_value
            yes_conf_score = conf_score_value
        if token_id in no_tokens:
            no_logit = logit_value
            no_conf_score = conf_score_value
    
    # Determine answer based on which token appeared
    if yes_logit is not None and no_logit is None:
        answer = 'yes'
    elif no_logit is not None and yes_logit is None:
        answer = 'no'
    elif yes_logit is not None and no_logit is not None:
        # Both appeared - yes is the answer
        answer = 'yes'
    else:
        # No explicit yes/no tokens found in logits, will parse from text response
        answer = 'uncertain'
    
    return {
        'answer': answer,
        'yes_logit': yes_logit or 0,
        'no_logit': no_logit or 0,
        'yes_conf_score': yes_conf_score or 0,
        'no_conf_score': no_conf_score or 0,
        'response_conf_score': sum(all_conf_scores) / len(all_conf_scores) if len(all_conf_scores) > 0 else 0,
    }


def get_ref_score(image_paths: List[str], prompt: str, reward_model):
    """Get reference image_reward score for a given images and prompts."""

    images = [
        load_image_rgb(image_path) for image_path in image_paths
    ]
    all_scores = reward_model.select(None, images, prompt, None, 4)

    assert all_scores.ndim == 1

    # print()
    try:
        top_return = torch.topk(all_scores, 4)
    except:
        print("Error in topk")
        print("all_scores shape is", all_scores.shape)
        print("all_scores is", all_scores)
        exit()
    top_scores, top_indices = top_return.values, top_return.indices

    return all_scores, top_scores, top_indices


def get_loss(model, ref_ids, images, image_embeds, **kwargs):
    (
        # inputs,
        # position_ids,
        # attention_mask,
        # _,
        # inputs_embeds,
        # _,
        # vision_tower_aux_feature_list,
        # vision_tower_aux_attention_masks_list,
        # final_vision_feature_size,
        # global_context_feature,


        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
        selected_features,
        input_embed_mask,
        attention_bias,
        extra_mm,


    ) = model.prepare_inputs_labels_for_multimodal(
        ref_ids,
        None,
        None,
        None,
        None,
        images=images,
        image_embeds=image_embeds,
    )

    outputs = model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=kwargs['use_cache'],
        return_dict=True,
        decoding=False,
        answer_token_mask=extra_mm,
        guidance_level=None,
    )

    return outputs


def cmd_t2i(args: argparse.Namespace, index: int, metadata: Dict, qa_dict: Dict,
            tokenizer, model, image_processor, reward_model, decoder_bundle, start_id: int, end_id: int, eos_id: int) -> None:
    """Generate text-to-image samples using geneval_eval_ddp.py style saving.
    
    Args:
        args: Command line arguments
        index: Sample index from metadata file
        metadata: Metadata dict containing 'prompt' key
        tokenizer: Loaded tokenizer
        model: Loaded model
        image_processor: Loaded image processor
        start_id: Start image token ID
        end_id: End image token ID
        eos_id: EOS token ID
    """
    # Setup output directories using geneval_eval_ddp.py structure
    judge_template = args.post_qa_template_path.split("/")[-1].split(".")[0]
    ckpt_name = args.model_path.split("/")[-1].replace(".", "-")
    exp_name = "_".join([str(args.scaling_rounds), judge_template, "scale", args.post_qa_mode])
    outpath = os.path.join(args.output_dir, ckpt_name, exp_name, f"{index:0>5}")
    sample_dir = os.path.join(outpath, "samples")
    
    prompt = metadata['prompt']
    assert prompt in qa_dict.keys(), f"Prompt {prompt} not found in qa_dict"

    questions = qa_dict[prompt]

    n_samples = args.n_samples * args.scaling_rounds
    # Check for resumption - skip if already completed
    metadata_path = os.path.join(outpath, "metadata.jsonl")
    if os.path.exists(metadata_path) and os.path.exists(sample_dir):
        existing_images = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
        if len(existing_images) >= n_samples:
            print(f"Skipping sample {index} - already has {len(existing_images)} images")
            return
    
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Count existing samples to continue from where we left off
    if os.path.exists(sample_dir):
        existing_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
        sample_count = len(existing_files)
    else:
        sample_count = 0
    
    if sample_count >= n_samples:
        print(f"Already have {sample_count} samples for index {index}")
        return

    prompt_text = metadata['prompt'].replace(IMAGE_PLACEHOLDER, "")
    prompt = build_prompt(prompt_text, model_config=model.config, with_image=False, gen_image=True)
    input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)
    
    # Generate args.n_samples images (in batches if needed)
    batch_size = getattr(args, 'batch_size', 1)
    remaining_samples = n_samples - sample_count

    text_responses = {}
    all_image_paths = []

    for _ in range((remaining_samples + batch_size - 1) // batch_size):
        if sample_count >= n_samples:
            break
            
        with torch.inference_mode():
            output_ids, image_embeds = model.generate(
                input_ids,
                images=None,
                **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens),
            )
        
        gen_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # generated_text = tokenizer.decode(output_ids[0] if isinstance(output_ids, list) else output_ids, skip_special_tokens=True)
        # all_generated_text.append(generated_text)
        
        if image_embeds is not None and image_embeds.numel() > 0:
            images = decode_image_embeds(model, image_embeds, decoder_bundle)
            image_paths = save_images(images, sample_dir, start_count=sample_count)
            sample_count += len(images)
        elif image_embeds is None or image_embeds.numel() == 0:
            # Create dummy image if generation failed
            print(f"WARNING: Model returned empty image embeds for index {index}. Saving dummy image.")
            dummy_image = Image.new('RGB', (1024, 1024), (0, 0, 0))
            image_paths = save_images([dummy_image], sample_dir, start_count=sample_count)
            sample_count += 1
        
        ## Potential post-step: QA using latent or decoded image

        post_qa_prompt = ""
        if args.post_qa_template_path:
            with open(args.post_qa_template_path, "r") as f:
                post_qa_prompt = f.read()
        elif args.post_qa_prompt:
            post_qa_prompt = args.post_qa_prompt
        
        judge_result = {}
        if post_qa_prompt:
            output_text = None
            score_result = None
            image_embeds = image_embeds.unsqueeze(0)

            yes_count = 0
            yes_logits = []
            no_logits = []
            yes_conf_scores = []
            no_conf_scores = []
            total_question_count = len(questions)
            all_input_questions = []
            all_responses = []

            if args.post_qa_mode == "latent":
                # prompt_post = build_prompt(post_qa_prompt, model_config=model.config, with_image=True, num_frames=1)
                # input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)
                
                question = f"Does this image align with the prompt: {prompt_text}?"
                prompt_post = build_prompt(question, model_config=model.config, with_image=True, gen_image=False, num_frames=1)
                input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)
                with torch.inference_mode():
                    # print("input_ids_post shape is", input_ids_post.shape)
                    # print("image_embeds shape is", image_embeds.shape)
                    
                    output_ids_post, _, output_logits, output_scores = model.generate(
                        input_ids_post,
                        image_embeds=image_embeds,
                        return_scores=True,
                        **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                    )

                output_text = tokenizer.decode(output_ids_post[0], skip_special_tokens=True)
                # Get yes/no confidence scores
                score_result = get_score(output_text, output_logits, output_scores, output_ids_post[0], tokenizer)
                all_responses.append(output_text)
                all_input_questions.append(prompt_post)

                ########################################################## get prompt loss ##########################################################

                ref_question = build_prompt("Describe this image.", model_config=model.config, with_image=True, gen_image=False, num_frames=1)
                ref_question_ids = tokenize_prompt(ref_question, tokenizer, device=model.device)
                ref_prompt_ids = tokenize_prompt(prompt_text, tokenizer, device=model.device)

                ref_ids = torch.cat([ref_question_ids, ref_prompt_ids], dim=1)

                with torch.inference_mode():
                    ref_outputs = get_loss(
                        model, ref_ids, None, image_embeds,
                        **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                    )

                    ref_logits = ref_outputs.logits[:, -ref_prompt_ids.shape[1]:, :]

                    # we get per token loss here
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    shift_logits = ref_logits[..., :-1, :]
                    shift_labels = ref_prompt_ids[..., 1:]
                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    per_token_ref_loss = loss_fct(shift_logits, shift_labels).view(ref_logits.shape[0], -1)
                    ref_loss = per_token_ref_loss.mean()

                    # we get confidence score here
                    probs = torch.softmax(ref_logits, dim=-1)
                    log_probs = torch.log(model.config.vocab_size * probs)

                    # averageing over both tokens and vocabs
                    conf_score = -log_probs.mean()
                
                ########################################################## get image confidence ##########################################################
                # for image confidence, we pass the image embeds with the generation prompt to the model and get the confidence score
                with torch.inference_mode():
                    gen_text_tokens = tokenize_prompt(gen_output_text + "<im_start><image><im_end>", tokenizer, device=model.device)
                    image_prompt_ids = torch.cat([input_ids, gen_text_tokens], dim=1)
                    ref_image_outputs = get_loss(
                        model, image_prompt_ids, None, image_embeds,
                        **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                    )

                    # output_ids corresponds to the gen_output_text
                    ref_image_logits = ref_image_outputs.logits[:, (input_ids.shape[1] + len(output_ids[0])):-1, :]

                    probs = torch.softmax(ref_image_logits, dim=-1)
                    log_probs = torch.log(model.config.vocab_size * probs)

                    # averageing over both tokens and vocabs
                    image_conf_score = -log_probs.mean()


                judge_result = {
                    'input_questions': all_input_questions,
                    'score': int(score_result['answer'] == 'yes'),
                    'gen_output_text': gen_output_text,
                    'responses': all_responses,
                    'prompt_loss': ref_loss.item(),
                    'prompt_conf_score': conf_score.item(),
                    'image_conf_score': image_conf_score.item(),
                    **score_result,
                }

            elif args.post_qa_mode == "image":
                assert image_paths is not None, "generated image not found"
                img_path = image_paths[0]
                image = load_image_rgb(img_path)
                images_tensor, size = preprocess_single_image(image, image_processor, device=model.device, dtype=torch.float32)
                # print("images_tensor shape is", images_tensor.shape, "dtype is", images_tensor.dtype)
                # prompt_post = build_prompt(post_qa_prompt, model_config=model.config, with_image=True, num_frames=1)
                # input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)

                for question in questions:
                    if question == "":
                        continue
                    question = f"Does this image align with the prompt: {prompt_text}?"
                    prompt_post = build_prompt(question, model_config=model.config, with_image=True, gen_image=False, num_frames=1)
                    input_ids_post = tokenize_prompt(prompt_post, tokenizer, device=model.device)
                    with torch.inference_mode():
                        output_ids_post, _, output_scores = model.generate(
                            input_ids_post,
                            images=images_tensor,
                            return_scores=True,
                            **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                        )
                    output_text = tokenizer.decode(output_ids_post[0], skip_special_tokens=True)
                    # Get yes/no confidence scores
                    score_result = get_score(output_text, output_scores, output_ids_post[0], tokenizer)
                    all_responses.append(output_text)
                    all_input_questions.append(prompt_post)

                    if score_result['answer'] == 'yes':
                        yes_count += 1
                    if score_result['yes_logit'] is not None:
                        yes_logits.append(score_result['yes_logit'])

                    if score_result['no_logit'] is not None:
                        no_logits.append(score_result['no_logit'])
                    break
                
                # get loss
                ref_question = build_prompt("Describe this image.", model_config=model.config, with_image=True, gen_image=False, num_frames=1)
                ref_question_ids = tokenize_prompt(ref_question, tokenizer, device=model.device)
                ref_prompt_ids = tokenize_prompt(prompt_text, tokenizer, device=model.device)

                ref_ids = torch.cat([ref_question_ids, ref_prompt_ids], dim=1)

                with torch.inference_mode():
                    ref_outputs = get_loss(
                        model, ref_ids, images_tensor, None,
                        **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.post_qa_max_new_tokens),
                    )

                    ref_logits = ref_outputs.logits[:, -ref_prompt_ids.shape[1]:, :]

                    # we get per token loss here
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    shift_logits = ref_logits[..., :-1, :]
                    shift_labels = ref_prompt_ids[..., 1:]
                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    per_token_ref_loss = loss_fct(shift_logits, shift_labels).view(ref_logits.shape[0], -1)
                    ref_loss = per_token_ref_loss.mean()

                judge_result = {
                    'score': yes_count / total_question_count,
                    'yes_logits': sum(yes_logits) / len(yes_logits) if len(yes_logits) > 0 else 0,
                    'no_logits': sum(no_logits) / len(no_logits) if len(no_logits) > 0 else 0,
                    'responses': all_responses,
                    'input_questions': all_input_questions,
                    'prompt_loss': ref_loss.item(),
                }
            else:
                raise SystemExit(f"Unknown post-qa mode: {args.post_qa_mode}")
        
        all_image_paths.append(image_paths[0])
        text_responses[sample_count - 1] = judge_result

    all_scores, top_scores, top_indices = get_ref_score(all_image_paths, prompt_text, reward_model)

    # Save metadata in geneval_eval_ddp.py format
    output_metadata = {
        **metadata,  # Include original metadata
        "model_path": args.model_path,
        "guidance_level": args.guidance_level,
        "max_new_tokens": args.max_new_tokens,
        "n_samples_generated": sample_count,
        # "generated_texts": all_generated_text,
        "input_prompt": prompt,
        "text_responses": text_responses,
        "all_ir_scores": all_scores.tolist(),
        "top_ir_scores": top_scores.tolist(),
        "top_ir_indices": top_indices.tolist(),
    }
    write_metadata(metadata_path, output_metadata)
    
    print(f"Sample {index}: Generated {sample_count} images in {sample_dir}")


def cmd_latent(args: argparse.Namespace, index: int, metadata: Dict,
               tokenizer, model, image_processor, start_id: int, end_id: int, eos_id: int) -> None:
    """Process latent tensors using geneval_eval_ddp.py style saving.
    
    Args:
        args: Command line arguments
        index: Sample index from metadata file
        metadata: Metadata dict
        tokenizer: Loaded tokenizer
        model: Loaded model
        image_processor: Loaded image processor
        start_id: Start image token ID
        end_id: End image token ID
        eos_id: EOS token ID
    """
    # Setup output directories
    outpath = os.path.join(args.output_dir, "samples", f"{index:0>5}")
    sample_dir = os.path.join(outpath, "samples")
    metadata_path = os.path.join(outpath, "metadata.jsonl")
    
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    try:
        image_embeds = torch.load(args.latent, map_location="cpu")
    except Exception:
        raise SystemExit(f"Failed to load latent from: {args.latent}")
    if not isinstance(image_embeds, torch.Tensor):
        raise SystemExit(f"Latent file is not a tensor: {args.latent}")
    image_embeds = image_embeds.to(device=model.device, dtype=model.dtype)

    sample_count = 0
    
    if args.action == "decode":
        if not args.skip_decoder:
            bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=getattr(args, 'decoder_repo', DEFAULT_DECODER_REPO))
            images = decode_image_embeds(model, image_embeds, bundle)
            image_paths = save_images(images, sample_dir, start_count=0)
            sample_count = len(images)
        
        output_metadata = {
            **metadata,
            "mode": "latent-decode",
            "model_path": args.model_path,
            "latent_path": args.latent,
            "skip_decoder": args.skip_decoder,
            "n_samples_generated": sample_count,
        }
        write_metadata(metadata_path, output_metadata)
        print(f"✓ Sample {index}: {sample_count} images decoded")
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
        
        output_metadata = {
            **metadata,
            "mode": f"latent-{args.action}",
            "model_path": args.model_path,
            "latent_path": args.latent,
            "prompt": args.prompt,
            "guidance_level": args.guidance_level,
            "output_text": output_text,
        }
        write_metadata(metadata_path, output_metadata)
        print(f"Sample {index}: {output_text}")
        return

    raise SystemExit(f"Unknown latent action: {args.action}")


def cmd_img(args: argparse.Namespace, index: int, metadata: Dict,
            tokenizer, model, image_processor, start_id: int, end_id: int, eos_id: int) -> None:
    """Generate images from input image using geneval_eval_ddp.py style saving.
    
    Args:
        args: Command line arguments
        index: Sample index from metadata file
        metadata: Metadata dict
        tokenizer: Loaded tokenizer
        model: Loaded model
        image_processor: Loaded image processor
        start_id: Start image token ID
        end_id: End image token ID
        eos_id: EOS token ID
    """
    # Setup output directories
    outpath = os.path.join(args.output_dir, "samples", f"{index:0>5}")
    sample_dir = os.path.join(outpath, "samples")
    metadata_path = os.path.join(outpath, "metadata.jsonl")

    n_samples = args.n_samples * args.scaling_rounds
    
    # Check for resumption
    if os.path.exists(metadata_path) and os.path.exists(sample_dir):
        existing_images = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
        if len(existing_images) >= n_samples:
            print(f"Skipping sample {index} - already has {len(existing_images)} images")
            return
    
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Count existing samples
    if os.path.exists(sample_dir):
        existing_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
        sample_count = len(existing_files)
    else:
        sample_count = 0
    
    if sample_count >= n_samples:
        print(f"Already have {sample_count} samples for index {index}")
        return

    image = load_image_rgb(args.image)
    images_tensor, size = preprocess_single_image(image, image_processor, device=model.device, dtype=model.dtype)

    prompt = build_prompt(metadata['prompt'], model_config=model.config, with_image=True, num_frames=1)
    input_ids = tokenize_prompt(prompt, tokenizer, device=model.device)

    all_generated_text = []
    batch_size = getattr(args, 'batch_size', 1)
    remaining_samples = n_samples - sample_count
    
    for _ in range((remaining_samples + batch_size - 1) // batch_size):
        if sample_count >= n_samples:
            break
            
        with torch.inference_mode():
            output_ids, image_embeds = model.generate(
                input_ids,
                images=images_tensor,
                **_common_gen_kwargs(start_id, end_id, eos_id, args.guidance_level, args.max_new_tokens),
            )
        
        generated_text = tokenizer.decode(output_ids[0] if isinstance(output_ids, list) else output_ids, skip_special_tokens=True)
        all_generated_text.append(generated_text)
        
        if not args.skip_decoder and image_embeds is not None and image_embeds.numel() > 0:
            bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=getattr(args, 'decoder_repo', DEFAULT_DECODER_REPO))
            images = decode_image_embeds(model, image_embeds, bundle)
            image_paths = save_images(images, sample_dir, start_count=sample_count)
            sample_count += len(images)

    output_metadata = {
        **metadata,
        "mode": "img",
        "model_path": args.model_path,
        "input_image": args.image,
        "guidance_level": args.guidance_level,
        "n_samples_generated": sample_count,
        "generated_texts": all_generated_text,
    }
    write_metadata(metadata_path, output_metadata)
    print(f"Sample {index}: Generated {sample_count} images in {sample_dir}")


def set_deterministic_seed(base_seed: int, rank: int = 0) -> None:
    """Set seeds and deterministic flags for reproducibility.
    Uses base_seed offset by rank for multi-process runs.
    """
    seed = int(base_seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scale-RAE inference CLI (t2i, latent, img)")
    sp = p.add_subparsers(dest="cmd", required=True)

    # t2i
    p_t2i = sp.add_parser("t2i", help="Text-to-image generation")
    p_t2i.add_argument("--seed", type=int, default=42)
    p_t2i.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_t2i.add_argument("--output-dir", default="outputs")
    p_t2i.add_argument("--guidance-level", type=float, default=1.0)
    p_t2i.add_argument("--skip-decoder", action="store_true")
    p_t2i.add_argument("--save-latent", action="store_true")
    p_t2i.add_argument("--max-new-tokens", type=int, default=512)
    p_t2i.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    p_t2i.add_argument("--n-samples", type=int, default=4, help="Number of samples to generate per prompt")
    p_t2i.add_argument("--metadata_file", type=str, required=True, help="Path to metadata jsonl file with prompts")
    p_t2i.add_argument("--scaling-rounds", type=int, default=1, help="Scaling rounds")
    p_t2i.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")

    p_t2i.add_argument("--post-qa-template-path", default=None)
    p_t2i.add_argument("--post-qa-prompt", default=None)
    p_t2i.add_argument("--post-qa-mode", choices=["latent", "image"], default="latent")
    p_t2i.add_argument("--post-qa-max-new-tokens", type=int, default=512)

    # latent
    p_lat = sp.add_parser("latent", help="Operate on saved latents (decode, qa, continue)")
    p_lat.add_argument("--seed", type=int, default=42)
    p_lat.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_lat.add_argument("--latent", required=True, help="Path to saved latent .pt")
    p_lat.add_argument("--action", required=True, choices=["decode", "qa", "continue"])
    p_lat.add_argument("--prompt", default=None, help="Prompt for qa/continue")
    p_lat.add_argument("--output-dir", default="outputs")
    p_lat.add_argument("--guidance-level", type=float, default=1.0)
    p_lat.add_argument("--skip-decoder", action="store_true")
    p_lat.add_argument("--max-new-tokens-qa", type=int, default=64)
    p_lat.add_argument("--n-samples", type=int, default=4, help="Number of samples to generate")
    p_lat.add_argument("--metadata_file", type=str, required=True, help="Path to metadata jsonl file")
    p_lat.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")

    # img
    p_img = sp.add_parser("img", help="Use an input image with a prompt")
    p_img.add_argument("--seed", type=int, default=42)
    p_img.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Model path or HF repo (default: {DEFAULT_MODEL_PATH})")
    p_img.add_argument("--image", required=True)
    p_img.add_argument("--output-dir", default="outputs")
    p_img.add_argument("--guidance-level", type=float, default=1.0)
    p_img.add_argument("--skip-decoder", action="store_true")
    p_img.add_argument("--save-latent", action="store_true")
    p_img.add_argument("--max-new-tokens", type=int, default=512)
    p_img.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    p_img.add_argument("--n-samples", type=int, default=4, help="Number of samples to generate per prompt")
    p_img.add_argument("--metadata_file", type=str, required=True, help="Path to metadata jsonl file with prompts")
    p_img.add_argument("--decoder-repo", default=DEFAULT_DECODER_REPO, help=f"Decoder HF repo (default: {DEFAULT_DECODER_REPO})")

    return p


def generate_worker(rank, world_size, mode, args, indexed_metadatas, qa_dict):
    """Worker function for multi-GPU processing.
    
    Args:
        rank: GPU rank (0, 1, 2, ...)
        world_size: Total number of GPUs
        mode: Command mode (t2i, latent, img)
        args: Command line arguments
        indexed_metadatas: List of (index, metadata) tuples
    """
    # Set deterministic seed for this worker
    set_deterministic_seed(args.seed, rank)
    
    # Set device for this worker
    if world_size > 1:
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
    else: 
        device = "cuda:0"
        torch.cuda.set_device(device)

    # Load model once for all samples (this is the key optimization!)
    # print(f"[GPU {rank}] Loading model...")
    tokenizer, model, image_processor, context_len = load_model(args.model_path, device=device)
    model.eval()
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    print(f"[GPU {rank}] Model loaded successfully")

    # Load reward model
    # print(f"[GPU {rank}] Loading reward model...")
    reward_model = ImageRewardSelector(args, device=device, debug=True)
    print(f"[GPU {rank}] Reward model loaded successfully")

    # Load decoder once for this worker
    # print(f"[GPU {rank}] Loading decoder...")
    decoder_bundle = build_decoder(model, model_path=args.model_path, decoder_repo_id=getattr(args, 'decoder_repo', DEFAULT_DECODER_REPO))
    print(f"[GPU {rank}] Decoder loaded successfully")

    # Split metadata across GPUs
    samples_for_this_gpu = indexed_metadatas[rank::world_size]
    print(f"[GPU {rank}] Processing {len(samples_for_this_gpu)} samples")
    
    # Process each sample
    for index, metadata in tqdm(samples_for_this_gpu, desc=f"GPU {rank}"):
        # try:
        if mode == "t2i":
            cmd_t2i(args, index, metadata, qa_dict, tokenizer, model, image_processor, reward_model, decoder_bundle, start_id, end_id, eos_id)
        elif mode == "latent":
            cmd_latent(args, index, metadata, qa_dict, tokenizer, model, image_processor, start_id, end_id, eos_id)
        elif mode == "img":
            cmd_img(args, index, metadata, qa_dict, tokenizer, model, image_processor, start_id, end_id, eos_id)
        else:
            raise SystemExit(f"Unknown command: {mode}")
        # except Exception as e:
        #     print(f"[GPU {rank}] Error processing sample {index}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     continue


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_deterministic_seed(args.seed)

    if not torch.cuda.is_available():
        print("Running on CPU (single process)")
        world_size = 0
    else:
        world_size = torch.cuda.device_count()

    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    indexed_metadatas = list(enumerate(metadatas))

    with open("assets/gpt_qa_dict.json", "r") as fp:
        qa_dict = json.load(fp)

    if world_size <= 1:
        print("Running on single GPU")
        set_deterministic_seed(args.seed, 0)
        # Load all metadata with indices
        
        generate_worker(0, 1, args.cmd, args, indexed_metadatas, qa_dict)
    else:
        print(f"Spawning {world_size} GPU workers...")
        mp.spawn(
            generate_worker,
            args=(world_size, args.cmd, args, indexed_metadatas, qa_dict),
            nprocs=world_size,
            join=True
        )
    
    print("✓ All tasks completed")


if __name__ == "__main__":
    main()


