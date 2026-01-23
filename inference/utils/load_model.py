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
    dtype: torch.dtype = torch.float16
):
    """
    Load the Scale-RAE model
    
    Args:
        model_path: Path or HF repo ID for Scale-RAE model
        model_base: Base model path if needed
        device: Device to load model on ('cuda' or 'cpu')
        dtype: Data type for model weights
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    # Disable torch init as done in original code
    disable_torch_init()

    # Get model name from path
    model_name = get_model_name_from_path(model_path)
    print(f"Loading Scale-RAE model: {model_path, model_name}")
   
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        if dtype == torch.float16:
            dtype = torch.float32
    
    # Load the model for single-GPU inference
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name, 
        device=device,
        device_map={"": device},  # Force single device (no multi-GPU)
        torch_dtype=dtype,
    )
    
    print(f"Model loaded successfully with context length: {context_len}")
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