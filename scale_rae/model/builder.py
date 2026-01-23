#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from scale_rae.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ezcolorlog import root_logger as logger

from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM


def load_pretrained_model(model_path, model_base=None, model_name="", 
                          load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", 
                          use_flash_attn=False, **kwargs):
    """
    Load pretrained Scale-RAE model.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        model_base: Unused (kept for API compatibility)
        model_name: Model identifier (for internal routing)
        load_8bit: Whether to load in 8-bit quantization
        load_4bit: Whether to load in 4-bit quantization
        device_map: Device mapping strategy
        device: Target device
        use_flash_attn: Whether to use Flash Attention 2
    
    Returns:
        tokenizer: Model tokenizer
        model: Loaded Scale-RAE model
        image_processor: List of image processors for vision towers
        context_len: Maximum context length
    """
    
    # Add prefix for internal routing if needed
    if model_name and "scale_rae" not in model_name.lower():
        model_name = "scale_rae_qwen2_" + model_name

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Load Scale-RAE Qwen2 model
    logger.info(f'Loading Scale-RAE model from {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Remove torch_dtype for compatibility with Scale-RAE
    if 'torch_dtype' in kwargs:
        kwargs['torch_dtype'] = None
    
    model = ScaleRAEQwenForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )

    # Load vision towers for multimodal processing
    vision_tower_aux_list = model.get_vision_tower_aux_list()
    for vision_tower_aux in vision_tower_aux_list:
        vision_tower_aux.load_model()
    image_processor = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]

    # Get context length from model config
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
