# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from scale_rae.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from scale_rae.conversation import conv_templates
from scale_rae.model.builder import load_pretrained_model
from scale_rae.utils import disable_torch_init
from scale_rae.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import torch
import os

def load_scale_rae_model(
    model_path: str = "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B",
    model_base: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    single_gpu: bool = False,
):
    """
    Load the Scale-RAE model with automatic multi-GPU support.
    
    Uses device_map="auto" to distribute model across available GPUs by default.
    For DDP/data-parallel use, set single_gpu=True to load on a specific device.
    
    Args:
        model_path: Path or HF repo ID for Scale-RAE model
        model_base: Base model path if needed
        device: Device to load model on ('cuda', 'cuda:0', 'cpu', etc.)
        dtype: Data type for model weights
        single_gpu: If True, load entire model on the specified device (for DDP)
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    print(f"Loading Scale-RAE model: {model_path}")
   
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        dtype = torch.float32
    
    # Determine device mapping strategy
    if single_gpu or not device.startswith("cuda"):
        # Single GPU mode: load entire model on specified device
        device_map = {"": device}
        print(f"Loading model on single device: {device}")
    else:
        # Multi-GPU mode: auto distribute
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPU(s) with auto device mapping")
        device_map = "auto"
    
    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name, 
        device=device if single_gpu else "cuda",
        device_map=device_map,
        torch_dtype=dtype,
    )
    
    # Consolidate diff_head onto single GPU to avoid cross-device errors during diffusion
    # Only needed for multi-GPU auto mode
    if device_map == "auto" and hasattr(model, 'diff_head'):
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            from accelerate.hooks import remove_hook_from_submodules
            diff_head_device = f"cuda:{n_gpus - 1}"
            
            remove_hook_from_submodules(model.diff_head)
            model.diff_head = model.diff_head.to(diff_head_device)
            
            if hasattr(model, 'diff_head_projector') and model.diff_head_projector is not None:
                remove_hook_from_submodules(model.diff_head_projector)
                model.diff_head_projector = model.diff_head_projector.to(diff_head_device)
            
            print(f"diff_head consolidated on {diff_head_device}")
    
    print(f"Model loaded (context_len={context_len})")
    return tokenizer, model, image_processor, context_len

if __name__ == "__main__":
    # Example usage
    # try read model_path from input
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B"
    tokenizer, model, image_processor, context_len = load_scale_rae_model(model_name)
    
    print(f"Model info:")
    print(f"- Context length: {context_len}")
    print(f"- Model device: {model.device}")
    print(f"- Model dtype: {model.dtype}")
    from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM
    model: ScaleRAEQwenForCausalLM
    # print(f"- Model diff_head : {model.diff_head.device}, {model.diff_head.dtype}")