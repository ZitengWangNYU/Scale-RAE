# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import re
import math
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch_xla.runtime as xr
import torch.distributed as dist

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import transformers
import tokenizers

import scale_rae

from scale_rae.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from scale_rae.train.scale_rae_trainer import ScaleRAETrainer

from scale_rae import conversation as conversation_lib

from scale_rae.utils import IS_XLA_AVAILABLE, process_video_with_decord
from scale_rae.mm_utils import tokenizer_image_token, tokenizer_image_token_llama3
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM
from PIL import Image

from ezcolorlog import root_logger as logger

from packaging import version


logger.setLevel(logging.INFO)

from safetensors.torch import load_file
from tabulate import tabulate


local_rank = None

XLA_DISABLE_FUNCTIONALIZATION = bool(os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))

PRINT_LOGS = True


def print_rank0(*args):
    if local_rank in (0, -1) and PRINT_LOGS:
        print(*args)


def debug_print(*args, **kwargs):
    """Debug print function controlled by SCALE_RAE_DEBUG environment variable."""
    if os.getenv("SCALE_RAE_DEBUG", "0") == "1":
        print(*args, **kwargs)


def log_rank0(log):
    if local_rank in (0, -1) and PRINT_LOGS:
        logger.info(log, stacklevel=2)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_adapter_and_vision_head: bool = field(default=False)
    tune_vision_head: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_aux_list: Optional[str] = field(default=None)
    generation_alignment_tower: Optional[str] = field(default=None, metadata={"help": "Path to the model used for the generative alignment target, e.g., a VAE."})
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_adapter_and_vision_head: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    normalize_vision: bool = field(default=True)
    vision_loss: Optional[str] = field(default="regression-loss")
    vision_loss_mode: Optional[str] = field(default="causal", metadata={"help": "Vision loss mode: 'causal', 'block', 'query', or 'ar-ddt'"})
    vision_coef: Optional[float] = field(default=1)   # default to the last layer
    dit_cls: Optional[str] = field(default="DiT", metadata={"help": "Backbone for diffusion head: 'DiT', 'DDT', or 'xattnDiT'."})
    # Optional auxiliary regression head alongside diffusion
    aux_regression: bool = field(default=False, metadata={"help": "Enable auxiliary regression head when using diffusion loss."})
    aux_regression_coef: float = field(default=1.0, metadata={"help": "Weight of auxiliary regression loss added to total loss."})
    # K-timestep tiling per sample for diffusion/DDT (query-mode training)
    diffusion_timesteps_per_sample: int = field(default=1, metadata={"help": "Number of timesteps per sample to tile within diffusion/DDT loss (query mode only)."})
    diffusion_split_per_token: Optional[int] = field(default=1)   # default to the last layer
    
    diffusion_model_hidden_size: Optional[int] = field(default=1152)   # default to the last layer
    diffusion_model_channels: Optional[int] = field(default=1152)   # default to the last layer


    diffusion_model_depth: Optional[int] = field(default=12)   # default to the last layer
    diffusion_model_heads: Optional[int] = field(default=16)   # default to the last layer

    diffusion_model_hidden_size_II: Optional[int] = field(default=0)   # default to the last layer
    diffusion_model_depth_II: Optional[int] = field(default=0)   # default to the last layer
    diffusion_model_heads_II: Optional[int] = field(default=0)   # default to the last layer

    diffusion_model_z_channels: Optional[int] = field(default=0)   # default to the last layer
    ddt_encoder_depth: Optional[int] = field(default=0)   # default to the last layer
    diffusion_class_dropout_prob: Optional[float] = field(default=0.0)   # default to the last layer
    diffusion_base_dim: Optional[int] = field(default=None, metadata={"help": "Override base dimension used to scale RF input (default 32*32*4)."})

    # NEW: Optional dataset stats path for input normalization in diffusion head
    diffusion_norm_stats_path: Optional[str] = field(default=None, metadata={"help": "Path to torch .pt with {'mean': Tensor[D], 'std' or 'var': Tensor[D]} for diffusion input normalization."})

    # NOTE: follow llava-onevision's setups
    si_token_len: int = 729 # token length (without newline) of per subimages for single image (si)
    miv_token_len: int = 196 # token length (without newline) for per subimages for multi images and video (miv)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    image_position: int = 35  # depends on v1 conv

    # make sure the batch size for image encoder is a constant
    # hold for both video and images
    max_images_per_sample: int = 1

    anyres_max_subimages: int = 1

    video_folder: str = ""
    video_fps: int = 1
    video_max_frames: int = 1
    video_force_sample: bool = False
    add_time_instruction: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_sampler_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    diff_head_lr: Optional[float] = None

    # NEW: keep diffusion-head LR constant after warm-up while the rest of the model follows the global schedule
    diff_head_constant_schedule: bool = field(
        default=False,
        metadata={"help": "If true, diffusion-head uses warm-up + constant LR instead of the global scheduler"},
    )

    # NEW: More flexible diffusion head LR scheduler types
    diff_head_lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Type of LR scheduler for diffusion head: 'cosine' (decay to 0), 'constant_with_warmup' (warmup then constant), 'cosine_with_min_lr' (decay to fraction of peak)"}
    )
    
    diff_head_min_lr_ratio: float = field(
        default=0.1,
        metadata={"help": "For cosine_with_min_lr scheduler: final LR as fraction of peak LR (e.g., 0.1 means decay to 10% of peak)"}
    )

    # Emergency rescue mode: load checkpoint and save shards without training
    rescue_ckpt: bool = field(
        default=False,
        metadata={"help": "Emergency rescue mode: load latest checkpoint, skip training, save as distributed shards"}
    )

    # sanity check arg
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "The total batch size for training. If passed, will be used to check that the "
                          "`per_device_train_batch_size` is set correctly."}
    )

    # Dataset prefiltering arguments
    prefilter_dataset: bool = field(
        default=False,
        metadata={"help": "Run dataset prefiltering mode instead of training"}
    )
    prefilter_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Output path for filtered dataset. Auto-generated if not provided."}
    )

    # GCSFS
    gcp_project: Optional[str] = field(default=None)
    """Can also set GCP_PROJECT environment variable."""
    gcs_output_dir: Optional[str] = field(default=None)
    """gs://<bucket>/<prefix>"""

    train_continue: bool = False
    load_weights: Optional[str] = ""
    resume_from_checkpoint: Optional[str] = ""

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.detach().cpu().clone() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_tower_aux', 'vision_resampler', 'vision_sampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    # output_dir = os.path.join('checkpoints', output_dir.split(os.sep)[-1])

    # Use the original path and append 'checkpoints' to it
    output_dir = os.path.join(output_dir, 'checkpoints')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', ]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "mm_projector")
        
        os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        ckpt = {
            'model': weight_to_save,
            'shard_metadata': trainer.model.get_shard_metadata()
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        debug_print(f'checkpoint saved to {ckpt_path}\n', end='')
        return

    # --- NEW: Save logic for tune_adapter_and_vision_head ---
    if False and getattr(trainer.args, "tune_adapter_and_vision_head", False):
        # Save Adapter and Vision Head
        keys_to_match = ['mm_projector', 'vision_head', 'diff_head', "latent_queries", 'embed_tokens']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "adapter_and_vision_head")
        
        os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        ckpt = {
            'model': weight_to_save,
            'shard_metadata': trainer.model.get_shard_metadata()
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        debug_print(f'checkpoint saved to {ckpt_path}\n', end='')
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    trainer._save(output_dir)
   
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    # for source in sources:
    #     for sentence in source:
    #         if DEFAULT_IMAGE_TOKEN in sentence['value']:
    #             sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
    #             sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
    #             sentence['value'] = sentence['value'].strip()
    #             if "mmtag" in conversation_lib.default_conversation.version:
    #                 sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
    #         replace_token = DEFAULT_IMAGE_TOKEN
    #         if data_args.mm_use_im_start_end:
    #             replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    #         sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()
        if prompt.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            prompt = prompt[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        conversations.append(prompt)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token_llama3(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = "<|eot_id|>"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split("<|eot_id|>")
        
        cur_len = 0

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            rou += sep
            
            # System Prompt
            if i == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # User Prompt
            elif i % 2 == 1:
                if i==1 and has_image:
                    round_len = len(tokenizer_image_token_llama3(rou, tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
                cur_len += round_len
            # Model Reponse
            elif i % 2 == 0:
                round_len = len(tokenizer(rou).input_ids)
                # Don't predict system prompt
                target[cur_len : cur_len + 3] = IGNORE_INDEX
                cur_len += round_len

            
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}


    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_phi3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.PHI3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            if i != 0: # remove the first \n token
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}, conversation is {conversation}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # copy from llava-video with slightly modification to fit transformers 4.37.0
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    
    # For qwen-2.5 3B

    special_tokens = tokenizer.additional_special_tokens_ids
    im_start, im_end = special_tokens[0], special_tokens[1]

    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id

            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def resize_and_pad_image(image, target_resolution, background_color=(0, 0, 0)):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), background_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 model_configs = None,
                 ):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_args = data_args
        self.model_configs = model_configs
        self.length = self._get_length()

        import torch_xla.core.xla_model as xm
        self.rank = xm.get_ordinal()

        self._build_offset_index()

    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, 'r') as file:
            for i, _ in enumerate(file):
                pass
        return i + 1

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.length

    def _build_offset_index(self):
        self.offsets = []
        with open(self.data_path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)
        self.length = len(self.offsets)



    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, 'length_list') and hasattr(self, 'modality_length_list'):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list

        self.length_list = []
        self.modality_length_list = []

        with open(self.data_path, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                assert not (self._has_image(sample) and self._has_video(sample)), "Video and image cannot exist in one single data sample"
                img_tokens = self.data_args.si_token_len if self._has_image(sample) or self._has_video(sample) else 0
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                if self._has_image(sample) or self._has_video(sample):
                    self.length_list.append(cur_len + img_tokens)
                modality_len = cur_len if 'image' in sample or 'video' in sample else -cur_len
                self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:
        return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']
    
    def _has_video(self, sample: dict) -> bool:
        return "video" in sample and not str(sample['video']) in ['', 'None', 'none', 'nan']

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # return self._getitem_(i)

        try:
            return self._getitem_(i)
        except BaseException as e:
            print(f"Error occurs when loading data at index {i}")
            print(e)
            import sys; sys.exit(-1)
    

    def check_image_tokens(self, data_dict, images, vision_token_len):
        """
        Drop‑in replacement that adds two stronger guarantees:

        ▸ uses correct length math  ( (vision_token_len‑1) × num_images )
        ▸ detects "half‑image" loss after right‑side truncation

        Assumptions left intact:
            • –200 remains the sole image‑placeholder ID
            • sequences are truncated from the RIGHT
        """
        input_ids   = data_dict["input_ids"]           # torch.Tensor
        max_length  = self.model_configs.tokenizer_model_max_length
        num_images  = len(images)
        placeholder = -200

        # ── 1️⃣  Placeholder count must match number of images
        ph_count = (input_ids == placeholder).sum().item()
        if ph_count != num_images:
            debug_print(f"[check‑1] Found {ph_count} placeholders for {num_images} images.")
            return False

        # ── 2️⃣  Full expansion (no truncation) must fit
        expanded_len = input_ids.numel() + (vision_token_len - 1) * num_images
        if expanded_len > max_length:
            debug_print(f"[check‑2] Expanded length would be {expanded_len} (> {max_length}).")
            return False

        # ── 3️⃣  If pre‑tokenizer truncation will occur, ensure no image is dropped/cut
        if input_ids.numel() > max_length:
            truncated_ids = input_ids[:max_length]              # right‑side policy
            kept_ph       = (truncated_ids == placeholder).sum().item()

            if kept_ph < ph_count:
                debug_print(f"[check‑3] Truncation would drop {ph_count - kept_ph} "
                    f"image placeholder(s).")
                return False

            # Even if all placeholders remain, their *expanded* form must still fit
            truncated_expanded_len = truncated_ids.numel() + (vision_token_len - 1) * num_images
            if truncated_expanded_len > max_length:
                debug_print(f"[check‑3] Truncation keeps placeholders, but expanded length "
                    f"would be {truncated_expanded_len} (> {max_length}).")
                return False

        # ── All good
        return True


    # def check_image_tokens(self, data_dict, images, vision_token_len):
    #     """
    #     Check that all images will be properly represented in the input.
        
    #     Args:
    #         data_dict: The dictionary containing input_ids and labels
    #         images: The list of images
    #         vision_token_len: The number of tokens per image
            
    #     Returns:
    #         bool: True if all checks pass, False if there are issues
    #     """
    #     # Check 1: Initial count of image tokens should match number of images
    #     count = torch.sum(data_dict["input_ids"].eq(-200)).item()
    #     if count != len(images):
    #         print(f"Bug check 1: Image token count mismatch! Found {count} tokens but have {len(images)} images")
    #         return False
            
    #     # Check 2: Would any image tokens be lost after truncation?
    #     max_length = self.model_configs.tokenizer_model_max_length

    #     # print("The length is, and sum is:",  len(data_dict['input_ids']), count)
    #     if len(data_dict['input_ids']) > max_length:
    #         # Create a copy for the check
    #         input_ids_truncated = data_dict['input_ids'][:max_length].clone()
    #         post_truncation_count = torch.sum(input_ids_truncated.eq(-200)).item()
    #         if post_truncation_count != len(images):
    #             print(f"Bug check 2: Truncation would lose image tokens! Pre: {count}, Post: {post_truncation_count}")
    #             return False
        
    #     # Check 3: Ensure context has enough space for all images at full resolution
    #     tokens_per_image = vision_token_len  
    #     total_image_tokens_needed = len(images) * tokens_per_image
    #     if len(data_dict['input_ids']) + total_image_tokens_needed > max_length:
    #         print(f"Bug check 3: Not enough context space for all image tokens at full resolution")
    #         return False
        
    #     # All checks passed
    #     return True

    def _getitem_(self, i):



        try:
                    # --- Use the persistent file handle instead of reopening each time ---
            with open(self.data_path, "rb") as f:
                f.seek(self.offsets[i])        # jump straight to the start of line *i*
                line = f.readline()            # read that one line
            sources = json.loads(line)         # no .strip() needed – readline() drops the '\n'

            dat = sources
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Sources should not be wrapped in a list"
            has_image = self._has_image(dat)
            has_video = self._has_video(dat)

            assert not (has_image and has_video), "Image and video should not appear in a single data sample"

            # Add image token if not present in the conversation
            if has_image or has_video:
                for source in sources:
                    if DEFAULT_IMAGE_TOKEN not in json.dumps(source['conversations']):
                        source['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + source['conversations'][0]['value']
            
            images = []
            vision_token_len = self.data_args.vision_tower_aux_token_len_list[0]

            if has_image:
                image_file = dat['image']

                # @MetaMorph Changes:
                # Need to cmake input image a list
                if not isinstance(image_file, list):
                    image_file = [image_file]                   



                image_folder = self.data_args.image_folder
                processor_aux_list = self.data_args.image_processor_aux_list

            
            
                try:
                    images = []
                    for image_path in image_file:
                        # Check if the path is absolute
                        if os.path.isabs(image_path):
                            # If absolute, load directly from the path
                            img_path = image_path
                        else:
                            # If relative, join with the image_folder
                            img_path = os.path.join(image_folder, image_path)
                        
                        # Open and convert the image
                        images.append(Image.open(img_path).convert('RGB'))

                    max_length = self.model_configs.tokenizer_model_max_length

                    if len(images)>(max_length//vision_token_len)-1:
                        debug_print("exceeded max length, skip it")
                        import random
                        return self.__getitem__(random.randint(0, len(self) - 1))
                        
                except:
                    logger.warning(f"Error occurs when load image from {image_file, image_path}")
                    import random
                    return self.__getitem__(random.randint(0, len(self) - 1))

                image_size = images[0].size

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                if self.data_args.image_aspect_ratio not in ['pad', 'anyres', 'square']:
                    raise NotImplementedError("Only pad and anyres are supported for now.")

                if self.data_args.image_aspect_ratio == 'pad':
                    image_aux_list = []
                    for processor_aux in processor_aux_list:
                        image_aux = image
                        target_resolution = processor_aux.crop_size['height']
                        image_aux = expand2square(image_aux, tuple(int(x*255) for x in processor_aux.image_mean)).resize((target_resolution, target_resolution))
                        image_aux = processor_aux.preprocess(image_aux, return_tensors='pt')['pixel_values'][0]
                        image_aux_list.append(image_aux)

                


                elif self.data_args.image_aspect_ratio == 'square':
                    image_aux_list = []
                    for image in images:
                        image_aux = processor_aux_list[0].preprocess(image, return_tensors='pt')['pixel_values'][0]
                        image_aux_list.append(image_aux)
                
                elif self.data_args.image_aspect_ratio == 'anyres':

                    image_aux_list = []
                    for processor_aux in processor_aux_list:

                        image_aux = image
                        target_resolution = processor_aux.crop_size['height']

                        # Choose resolutions where number of subimages <= anyres_max_subimages
                        possible_resolutions = [
                            (int(width * target_resolution), int(height * target_resolution))
                            for width in range(1, self.data_args.anyres_max_subimages + 1)
                            for height in range(1, self.data_args.anyres_max_subimages + 1)
                            if (width * height) <= self.data_args.anyres_max_subimages
                        ]

                        best_resolution = select_best_resolution(image.size, possible_resolutions)
                        # Use zero padding for anyres images
                        image_aux_padded = resize_and_pad_image(image, best_resolution)

                        patches = divide_to_patches(image_aux_padded, target_resolution)

                        # NOTE: we should use expand2square and then resize but we choose to use the following code to make sure our codebase aligns well with llava-onevision
                        image_aux = image_aux.resize((target_resolution, target_resolution))

                        image_patches = [image_aux] + patches
                        image_patches = [processor_aux.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in image_patches]
                        image_aux_list.append(torch.stack(image_patches))

                else:
                    raise NotImplementedError("Only pad, anyres and square are supported for now.")

                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)

            elif has_video:
                video_file = dat['video']
                video_folder = self.data_args.video_folder
                video_file = os.path.join(video_folder, video_file)
                
                try:
                    if os.path.isdir(video_file):
                        
                        if "shareVideoGPTV" in video_file: # shareVideoGPTV use 2FPS
                            avg_fps = 2
                        elif "TVQA" in video_file: # TVQA use 3FPS
                            avg_fps = 3
                        else: # for unknown video frames, we assume it is 1FPS
                            avg_fps = 1

                        frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                        frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
                        
                        video_time = len(frame_files) / avg_fps

                        if 'start' in dat:
                            start_time = float(dat['start'])
                            end_time = float(dat['end'])
                            start_frame = int(start_time * avg_fps)
                            end_frame = int(end_time * avg_fps)
                            end_frame = min(len(frame_files) - 1, end_frame)
                            frame_files = frame_files[start_frame:end_frame+1] # from start to end
                            video_time = end_time - start_time

                        frame_idx = [i for i in range(0, len(frame_files), avg_fps)]
                        frame_time = [i/avg_fps for i in frame_idx]

                        if self.data_args.video_max_frames > 0:
                            if len(frame_files) > self.data_args.video_max_frames or self.data_args.video_force_sample:
                                frame_idx = np.linspace(0, len(frame_files) - 1, self.data_args.video_max_frames, dtype=int).tolist()
                                frame_time = [i/avg_fps for i in frame_idx]


                        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                        # Read and store the sampled frames
                        num_frames_to_sample = len(frame_idx)
                        video = []
                        for idx in frame_idx:
                            frame_path = frame_files[idx]
                            try:
                                with Image.open(frame_path) as img:
                                    frame = img.convert("RGB")
                                    video.append(np.array(frame))
                            except IOError:
                                debug_print(f"Failed to read frame at path: {frame_path}")
                        video = np.stack(video)
                    elif video_file.endswith(".gif"):
                        if not os.path.exists(video_file):
                            debug_print("File {} not exist!".format(video_file))
                            raise FileNotFoundError
                        assert "start" not in dat and "end" not in dat, "start and end should not be in gif video"
                        assert "start_frame" not in dat and "end_frame" not in dat, "start_frame and end_frame should not be in gif video"
                        video, video_time, frame_time, num_frames_to_sample = process_gif_with_imageio(video_file, self.data_args)
                    else:
                        if not os.path.exists(video_file):
                            debug_print("File {} not exist!".format(video_file))
                            raise FileNotFoundError

                        if 'start_frame' in dat:
                            start_frame = dat['start_frame']
                            end_frame = dat['end_frame']
                            current_observation_frame = dat.get('current_observation_frame', None)

                            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_byframe(video_file, self.data_args, start_frame, end_frame, current_observation_frame)
                            if not video.size > 0:
                                raise ValueError(f"Video {video_file} is empty")
                        elif 'start' in dat:
                            start_time = dat['start']
                            end_time = dat['end']
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_bytime(video_file, self.data_args, start_time, end_time)
                        else:
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)
                except BaseException as error:
                    logger.warning(f"Error occurs when load video from {video_file}: {error}")
                    import random
                    return self.__getitem__(random.randint(0, len(self) - 1)) # if error occurs, random return another sample

                video_h, video_w = video.shape[1:3]
                image_size = (video_w, video_h)

                processor_aux_list = self.data_args.image_processor_aux_list
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                if self.data_args.image_aspect_ratio not in ['pad', 'anyres', 'square']:
                    raise NotImplementedError("Only pad and anyres are supported for now.")

                # Video always use pad
                image_aux_list = []
                if self.data_args.image_aspect_ratio in ['anyres', 'pad']:
                    for processor_aux in processor_aux_list:
                        target_resolution = processor_aux.crop_size['height']
                        frames = [expand2square(Image.fromarray(video[_], mode="RGB"), tuple(int(x*255) for x in processor_aux.image_mean)) for _ in range(video.shape[0])]
                        # processed_frames = [processor_aux.preprocess(frame, return_tensors='pt')['pixel_values'][0] for frame in frames]
                        # image_aux_list.append(torch.stack(processed_frames))
                        processed_frames = processor_aux.preprocess(frames, return_tensors='pt')['pixel_values']
                        image_aux_list.append(processed_frames)
                elif self.data_args.image_aspect_ratio in ['square']:
                    for processor_aux in processor_aux_list:
                        target_resolution = processor_aux.crop_size['height']
                        frames = [expand2square(Image.fromarray(video[_], mode="RGB"), tuple(int(x*255) for x in processor_aux.image_mean)) for _ in range(video.shape[0])]
                        processed_frames = processor_aux.preprocess(frames, return_tensors='pt')['pixel_values']
                        image_aux_list.append(processed_frames)
                else:
                    raise NotImplementedError("Only pad and anyres are supported for now.")
                
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])

            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=has_image or has_video)
            
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
            
            if has_image and not self.check_image_tokens(data_dict, images, vision_token_len):
                return self.__getitem__((i + 1) % self.__len__())


            if (data_dict['labels']!=IGNORE_INDEX).sum()==0:
                logger.warning("All tokens are masked, random return another sample")
                import random
                return self.__getitem__(random.randint(0, len(self) - 1)) # if all tokens are masked, random return another sample


            assert self.data_args.si_token_len >= 0
            assert self.data_args.miv_token_len >= 0
            
            si_token_len = self.data_args.si_token_len
            si_token_hws = int(math.sqrt(si_token_len))
            si_token_len_w_newline = si_token_len + si_token_hws

            miv_token_len = self.data_args.miv_token_len
            miv_token_hws = int(math.sqrt(miv_token_len))
            miv_token_len_w_newline = miv_token_len + miv_token_hws


            # Get token dimensions - simple square case, just use vision_token_len directly
            
            tokens_per_image = vision_token_len


            processor_aux_list = self.data_args.image_processor_aux_list
            processor_aux = processor_aux_list[0]

            # Simple square mode processing for interleaved images - NO newlines
            if has_image and self.data_args.image_aspect_ratio == 'square':
                
                # Process all images in the data
                n_imgs = len(image_aux_list)
                
                # Pad image tensors to fixed shape for TPU
                image_aux_list_padded = []
                
                image_aux_padded = torch.zeros(self.data_args.max_images_per_sample,  3, processor_aux.crop_size['height'], processor_aux.crop_size['width'])
                
                for i, image in enumerate(image_aux_list):
                    image_aux_padded[i] = image
                    # image_aux_list_padded.append(image_aux_padded)
                
                data_dict['image_aux_list'] = [image_aux_padded]
                
                # No newlines in simple square mode
                
                # Find all image token positions in the input
                input_ids = data_dict['input_ids']
                labels = data_dict['labels']
                img_token_positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                
                # Calculate total tokens needed for all images
                max_imgs = min(len(img_token_positions), self.data_args.max_images_per_sample)
                img_tokens_total = max_imgs * tokens_per_image
                
                # tokens_per_image is already vision_token_len
                T       = self.data_args.max_images_per_sample * tokens_per_image   # fixed length
                used    = max_imgs * tokens_per_image                               # real tokens
                PAD_VAL = T + 1                                                     # sentinel

                # 1. start fully padded
                vision_token_indices = torch.full((T,), PAD_VAL, dtype=torch.long)

                # 2. overwrite the real part (0 … used‑1) with raster indices
                if used:
                    vision_token_indices[:used] = torch.arange(used, dtype=torch.long)

                # 3. permutation vector (same length, guaranteed)
                data_dict["vision_token_indices"] = vision_token_indices.sort()[1]


                
                # Reconstruct the input_ids and labels with multiple image positions
                new_input_ids = []
                new_labels = []
                
                # Handle interleaved images by replacing each image token
                last_pos = 0
                for i, pos in enumerate(img_token_positions[:max_imgs]):
                    # Add text before this image
                    new_input_ids.append(input_ids[last_pos:pos])
                    new_labels.append(labels[last_pos:pos])
                    
                    # Simple fixed token count per image
                    img_tokens = torch.full((tokens_per_image,), IMAGE_TOKEN_INDEX, 
                                        dtype=input_ids.dtype, device=input_ids.device)
                    img_labels = torch.full((tokens_per_image,), IGNORE_INDEX, 
                                        dtype=labels.dtype, device=labels.device)
                    
                    new_input_ids.append(img_tokens)
                    new_labels.append(img_labels)
                    
                    # Update position tracker
                    last_pos = pos + 1
                
                # Add any remaining text after the last image
                if last_pos < len(input_ids):
                    new_input_ids.append(input_ids[last_pos:])
                    new_labels.append(labels[last_pos:])
                
                # Concatenate all parts
                data_dict['input_ids'] = torch.cat(new_input_ids)
                data_dict['labels'] = torch.cat(new_labels)
                
                # Create pseudo image tokens for padding to fixed length
                pseudo_img_tokens = torch.zeros(((self.data_args.max_images_per_sample - n_imgs)*tokens_per_image,), 
                                            dtype=input_ids.dtype, device=input_ids.device) + IMAGE_TOKEN_INDEX

                data_dict['pseudo_img_tokens'] = pseudo_img_tokens
                
                # Truncate input_ids and labels if needed to fit model_max_length
                max_length = self.model_configs.tokenizer_model_max_length
                if len(data_dict['input_ids']) > max_length:
                    data_dict['input_ids'] = data_dict['input_ids'][:max_length]
                    data_dict['labels'] = data_dict['labels'][:max_length]
                
                # Save image size
                data_dict['image_size'] = image_size



            # image exist in the data
            elif has_image:
                n_imgs = image_aux_list[0].size(0)

                image_aux_list_padded = []
                for image_aux in image_aux_list:
                    assert image_aux.shape[0] == n_imgs
                    image_aux_padded = torch.zeros((self.data_args.max_images_per_sample, *image_aux.size()[1:]))
                    image_aux_padded[:n_imgs] = image_aux
                    image_aux_list_padded.append(image_aux_padded)

                data_dict['image_aux_list'] = image_aux_list_padded

                num_img_patches = (best_resolution[0] // target_resolution, best_resolution[1] // target_resolution)

                # Calculate the unpadded feature shape and output feature shape
                image_w, image_h = image_size
                original_aspect_ratio = image_w / image_h
                padded_feature_w, padded_feature_h = (best_resolution[0] // target_resolution * si_token_hws, best_resolution[1] // target_resolution * si_token_hws)
                padded_feautre_aspect_ratio = padded_feature_w / padded_feature_h

                # Determine padding size and direction
                if original_aspect_ratio > padded_feautre_aspect_ratio:
                    # Padding was added to the height
                    scale_factor = padded_feature_w / image_w
                    unpadded_feature_w = padded_feature_w
                    unpadded_feature_h = int(image_h * scale_factor)
                    padding_feature_w = 0
                    padding_feature_h = (padded_feature_h - unpadded_feature_h) // 2
                else:
                    # Padding was added to the width
                    scale_factor = padded_feature_h / image_h
                    unpadded_feature_h = padded_feature_h
                    unpadded_feature_w = int(image_w * scale_factor)
                    padding_feature_h = 0
                    padding_feature_w = (padded_feature_w - unpadded_feature_w) // 2

                output_feature_shape = (unpadded_feature_h, unpadded_feature_w)

                # Create image token indexing
                num_img_tokens_total = self.data_args.max_images_per_sample * (miv_token_len_w_newline + si_token_len_w_newline)

                # Multi-image and video token indices
                miv_token_indices = torch.zeros(self.data_args.max_images_per_sample * miv_token_len_w_newline) + num_img_tokens_total + 1

                # Snapshot image token indices (remove newline for snapshot)
                snapshot_token_indices = torch.linspace(0, si_token_len_w_newline - 1, si_token_len_w_newline).reshape(si_token_hws, si_token_hws + 1).long() + miv_token_indices.numel()
                # NOTE: follow llava-video to remove the newline for snapshot
                snapshot_token_indices[:, -1] = num_img_tokens_total + 1

                # Anyres image token masks
                anyres_token_masks = torch.zeros(num_img_patches[1] * si_token_hws, num_img_patches[0] * si_token_hws).bool()
                if padding_feature_h > 0:
                    anyres_token_masks[:padding_feature_h, :] = True
                    anyres_token_masks[padding_feature_h+output_feature_shape[0]:, :] = True
                if padding_feature_w > 0:
                    anyres_token_masks[:, :padding_feature_w] = True
                    anyres_token_masks[:, padding_feature_w+output_feature_shape[1]:] = True
                # Reshape anyres masks: (nh, h, nw, w) -> (nh, nw, h, w) -> (n, h, w)
                anyres_token_masks = anyres_token_masks.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], si_token_hws).permute(0, 2, 1, 3).reshape(n_imgs - 1, si_token_hws, si_token_hws)

                anyres_token_masks_w_newline = torch.zeros(num_img_patches[1] * num_img_patches[0], si_token_hws,  si_token_hws + 1).bool()
                # Pad anyres images with newline token and mask all newline tokens            
                for index in range(n_imgs - 1):
                    if (index + 1) % num_img_patches[0] == 0: # if is end of line, then unmask it
                        ...
                    else:
                        anyres_token_masks_w_newline[index, :, -1] = True # mask
                anyres_token_masks_w_newline[:, :, :-1] = anyres_token_masks

                anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(num_img_patches[1], num_img_patches[0], si_token_hws, si_token_hws + 1).permute(0, 2, 1, 3).reshape(num_img_patches[1] * si_token_hws, num_img_patches[0] * (si_token_hws + 1)) # (nh * nw, h, w + 1) -> (nh, nw, h, w + 1) -> (nh * h, nw * (w + 1))
                if padding_feature_h > 0:
                    anyres_token_masks_w_newline[:padding_feature_h, :] = True
                    anyres_token_masks_w_newline[padding_feature_h+output_feature_shape[0]:, :] = True
                anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], (si_token_hws + 1)).permute(0, 2, 1, 3).reshape(num_img_patches[1] * num_img_patches[0], si_token_hws, (si_token_hws + 1))
                anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(-1)

                anyres_token_indices = torch.linspace(0, (n_imgs - 1) * si_token_len_w_newline - 1, (n_imgs - 1) * si_token_len_w_newline).long() + miv_token_indices.numel() + snapshot_token_indices.numel()
                anyres_token_indices = anyres_token_indices.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], (si_token_hws + 1)).permute(0, 2, 1, 3).reshape(num_img_patches[1] * num_img_patches[0], si_token_hws, (si_token_hws + 1)).flatten()
                anyres_token_indices = torch.where(anyres_token_masks_w_newline, num_img_tokens_total + 1, anyres_token_indices)

                padding_token_indices = torch.zeros((self.data_args.max_images_per_sample - n_imgs, si_token_hws, si_token_hws + 1)).long() + num_img_tokens_total + 1

                vision_token_indices = torch.cat([miv_token_indices.contiguous().view(-1), snapshot_token_indices.contiguous().view(-1), anyres_token_indices.contiguous().view(-1), padding_token_indices.contiguous().view(-1)], dim=0)

                num_real_img_tokens = (vision_token_indices != num_img_tokens_total + 1).sum()
                data_dict['vision_token_indices'] = vision_token_indices.view(-1).sort()[1]

                # rebuild the input_ids
                input_ids = data_dict['input_ids']
                labels = data_dict['labels']

                img_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
                pre_img_tokens, post_img_tokens = input_ids[:img_token_indices[0]], input_ids[img_token_indices[0]+1:]
                real_img_tokens = torch.zeros((num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
                pseudo_img_tokens = torch.zeros((self.model_configs.tokenizer_model_max_length - num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX

                data_dict['input_ids'] = torch.cat([pre_img_tokens, real_img_tokens, post_img_tokens])
                data_dict['pseudo_img_tokens'] = pseudo_img_tokens

                pre_img_labels, post_img_labels = labels[:img_token_indices[0]], labels[img_token_indices[0]+1:]
                real_img_labels = torch.zeros((num_real_img_tokens,)).long() + IGNORE_INDEX
                data_dict['labels'] = torch.cat([pre_img_labels, real_img_labels, post_img_labels])

            elif has_video:
                n_imgs = image_aux_list[0].size(0)

                image_aux_list_padded = []
                for image_aux in image_aux_list:
                    assert image_aux.shape[0] == n_imgs
                    image_aux_padded = torch.zeros((self.data_args.video_max_frames, *image_aux.size()[1:]))
                    image_aux_padded[:n_imgs] = image_aux
                    image_aux_list_padded.append(image_aux_padded)

                data_dict['image_aux_list'] = image_aux_list_padded

                assert [_.size(0) == self.data_args.video_max_frames for _ in image_aux_list]

                num_img_tokens_total = self.data_args.video_max_frames * miv_token_len
                vision_token_indices = torch.linspace(0, num_img_tokens_total - 1, num_img_tokens_total).long()

                vision_token_indices[n_imgs * miv_token_len:] = num_img_tokens_total + 1
                data_dict['vision_token_indices'] = vision_token_indices.view(-1).sort()[1]
                num_real_img_tokens = (vision_token_indices != num_img_tokens_total + 1).sum()

                # rebuild the input_ids
                input_ids = data_dict['input_ids']
                labels = data_dict['labels']
                
                img_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]

                assert img_token_indices.numel() == 1, "Only one image token should be there"

                pre_img_tokens, post_img_tokens = input_ids[:img_token_indices[0]], input_ids[img_token_indices[0]+1:]
                real_img_tokens = torch.zeros((num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
                
                data_dict['input_ids'] = torch.cat([pre_img_tokens, real_img_tokens, post_img_tokens])
                pseudo_img_tokens = torch.zeros((self.model_configs.tokenizer_model_max_length - num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
                data_dict['pseudo_img_tokens'] = pseudo_img_tokens

                pre_img_labels, post_img_labels = labels[:img_token_indices[0]], labels[img_token_indices[0]+1:]
                real_img_labels = torch.zeros((num_real_img_tokens,)).long() + IGNORE_INDEX
                data_dict['labels'] = torch.cat([pre_img_labels, real_img_labels, post_img_labels])

            elif self.data_args.is_multimodal:

                # image does not exist in the data, but the model is multimodal
                crop_size = 336
                processor_aux_list = self.data_args.image_processor_aux_list
                
                image_aux_list = []

                for processor_aux in processor_aux_list:
                    if self.data_args.max_images_per_sample > 0:
                        image_aux = torch.zeros(self.data_args.max_images_per_sample, 3, processor_aux.crop_size['height'], processor_aux.crop_size['width'])
                    else:
                        raise NotImplementedError

                    image_aux_list.append(image_aux)

                # same constants
                T       = self.data_args.max_images_per_sample * tokens_per_image
                PAD_VAL = T + 1
                # everything is padding because used = 0
                vision_token_indices = torch.full((T,), PAD_VAL, dtype=torch.long)

                data_dict["vision_token_indices"] = vision_token_indices.sort()[1]

                data_dict['pseudo_img_tokens'] = torch.zeros(self.data_args.max_images_per_sample * tokens_per_image).long() + IMAGE_TOKEN_INDEX

                image_size = (crop_size, crop_size)
                data_dict['image_aux_list'] = image_aux_list

                


            data_dict['image_size'] = image_size
            data_dict['has_video'] = has_video
            data_dict['has_image'] = has_image

            return data_dict
            
        except Exception as e:
            debug_print(f"another unsolved mysterious bug!!! {e}")
            return self.__getitem__((i + 1) % self.__len__())






def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0

def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = torch.ones(num_tokens_per_side, num_tokens_per_side+1, dtype=torch.bool)
    else:
        attention_mask = torch.ones(num_tokens_per_side, num_tokens_per_side, dtype=torch.bool)
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset((num_tokens_per_side, num_tokens_per_side), image_size)
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset-1:-1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :]=0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0)-1
    return attention_mask, position_ids

    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.

    Besides the usual padding duties this collator also pre-computes three
    fixed-shape helpers that the model consumes at run-time:

        answer_token_mask : (B , Lmax)  – 1 at IMAGE_TOKENs that belong to an
                                           answer image, else 0
        answer_img_mask   : (B , Mmax)  – 1 where the image slot is an answer
                                           image, else 0
        reverse_vti       : (B , Tmax)  – ready for apply_custom_kernel so
                                           that patch-rows originating from
                                           real images pick their sequence
                                           position, others keep default

    All three tensors have compile-time constant shapes which guarantees the
    TPU graph is re-used without recompilation.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_images_per_sample: int = 0
    tokens_per_image: int = 0

    video_max_frames: int = 0
    miv_token_len: int = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        pseudo_img_tokens = [_['pseudo_img_tokens'] for _ in instances]
        vision_token_indices = [_['vision_token_indices'] for _ in instances]

        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side

        if padding_side == "left":
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), 'constant', IGNORE_INDEX) for t in labels]
            pseudo_img_tokens = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), 'constant', self.tokenizer.pad_token_id) for t in pseudo_img_tokens]
        else:
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', IGNORE_INDEX) for t in labels]
            pseudo_img_tokens = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', self.tokenizer.pad_token_id) for t in pseudo_img_tokens]

        token_indices = []
        for _input_ids, _vision_token_indices in zip(input_ids, vision_token_indices):
            # _token_indices = torch.arange(max_length + num_img_tokens_total)
            _token_indices = torch.arange(max_length)
            num_img_tokens = (_input_ids == IMAGE_TOKEN_INDEX).sum()
            # _token_indices[max_length:max_length + num_img_tokens] = _token_indices[:max_length][_input_ids == IMAGE_TOKEN_INDEX]
            # _token_indices[:max_length][_input_ids == IMAGE_TOKEN_INDEX] = _vision_token_indices[:num_img_tokens] + max_length
            _token_indices[_input_ids == IMAGE_TOKEN_INDEX] = _vision_token_indices[:num_img_tokens] + max_length
            token_indices.append(_token_indices)


        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        pseudo_img_tokens = torch.stack(pseudo_img_tokens)

        vision_token_indices = torch.stack(token_indices)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            vision_token_indices=vision_token_indices,
        )

        if 'image_aux_list' in instances[0]:


            # image_aux_list = [instance['image_aux_list'] for instance in instances]
            # image_aux_list = [list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)]
            # if all(x is not None and x.shape == image_aux_list[0][0].shape for x in image_aux_list[0]):
            #     batch["images"] = [torch.cat(image_aux, dim=0) for image_aux in image_aux_list][0]
            #     # print(type(batch["images"]))
            #     batch['images_2'] = torch.empty(0, dtype=batch["images"].dtype, device=batch["images"].device)
                
            #     assert batch['images'].shape[0] == self.max_images_per_sample * input_ids.size(0)
            # else:
            #     raise NotImplementedError


            image_aux_list = [instance['image_aux_list'] for instance in instances]
            image_aux_list = [list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)]



            if all(x is not None and x.shape == image_aux_list[0][0].shape for x in image_aux_list[0]):
                batch["images"] = [torch.cat(image_aux, dim=0) for image_aux in image_aux_list][0]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Add image_gen_list for generation alignment (VAE training)
        if 'image_gen_list' in instances[0]:
            image_gen_list = [instance['image_gen_list'] for instance in instances]
            image_gen_list = [list(batch_image_gen) for batch_image_gen in zip(*image_gen_list)]
            
            # Check if image_gen_list has any elements before accessing
            if len(image_gen_list) > 0 and len(image_gen_list[0]) > 0:
                if all(x is not None and x.shape == image_gen_list[0][0].shape for x in image_gen_list[0]):
                    batch["images_gen"] = [torch.cat(image_gen, dim=0) for image_gen in image_gen_list][0]
                else:
                    raise NotImplementedError
            else:
                # If no images in batch, create an empty tensor with proper shape
                # This allows the batch to proceed without errors
                batch["images_gen"] = torch.empty(0, dtype=torch.float32)            

        # NOTE: this is temporally, should support both image and video later

        # ------------------------------------------------------------
        # CONSTANTS that must stay fixed for every batch on TPU
        # ------------------------------------------------------------
        Lmax = max_length                               # fixed text length
        Mmax = self.max_images_per_sample               # fixed image slots
        P    = self.tokens_per_image                    # tokens per image
        Tmax = Mmax * P

        # ------------------------------------------------------------------
        # Build answer_token_mask, answer_img_mask, reverse_vti (CPU side)
        # ------------------------------------------------------------------
        answer_tok_mask_list = []   # (Lmax,)
        answer_img_mask_list = []   # (Mmax,)
        reverse_vti_list     = []   # (Tmax,)

        for ids, lab, vti in zip(input_ids, labels, vision_token_indices):
        
            
            # 1. ans_img_mask: init with False, then check each image
            ans_img_mask = torch.zeros(Mmax, dtype=torch.bool)
            
            # Find all IMAGE_TOKEN positions
            img_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1)
            n_img_tokens = img_positions.numel()
            n_images = n_img_tokens // P  # number of actual images

            # For each image, check if the token before <image_start> is not IGNORE_INDEX
            for img_idx in range(n_images):
                first_token_pos = img_positions[img_idx * P]  # first token of this image
                if first_token_pos > 0:  # make sure we don't go out of bounds
                    label_before = lab[first_token_pos - 1]
                    if label_before != IGNORE_INDEX:
                        ans_img_mask[img_idx] = True
            
            # 2. ans_tok_mask: expand ans_img_mask to token level
            ans_tok_mask = torch.zeros(Lmax, dtype=torch.bool)
            for img_idx in range(n_images):
                if ans_img_mask[img_idx]:  # if this is an answer image
                    # Mark all P tokens of this image as answer tokens
                    start_idx = img_idx * P
                    end_idx = start_idx + P
                    token_positions = img_positions[start_idx:end_idx]
                    ans_tok_mask[token_positions] = True
            
            # 3. reverse_vti: init with 0 to Tmax-1, then map answer image patches
            #    back to their positions in the text sequence so that
            #    apply_custom_kernel(image_feats, input_embeds, reverse_vti)
            #    yields identity.
            reverse_vti = torch.arange(Tmax, dtype=torch.long, device=ids.device)

            if n_img_tokens:
                # Translate token positions -> patch-row indices using vti.
                patch_rows = (vti[img_positions] - Lmax)  # (n_img_tokens,)
                reverse_vti[patch_rows] = Tmax + img_positions

            answer_tok_mask_list.append(ans_tok_mask)
            answer_img_mask_list.append(ans_img_mask)
            reverse_vti_list.append(reverse_vti)

        answer_token_mask = torch.stack(answer_tok_mask_list)  # (B,Lmax)
        answer_img_mask   = torch.stack(answer_img_mask_list)  # (B,Mmax)
        reverse_vti       = torch.stack(reverse_vti_list)      # (B,Tmax)

        batch['answer_token_mask'] = answer_token_mask
        batch['answer_img_mask']   = answer_img_mask
        batch['reverse_vti']       = reverse_vti

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_configs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    # Check if WebDataset manifest
    if data_args.data_path.endswith('.json') and 'wds' in os.path.basename(data_args.data_path):
        # Use HuggingFace IterableDataset for WebDataset compatibility
        from .webdataset_trainer import WebDatasetLazySupervisedDataset
        print("================================================")
        print("Using WebDatasetLazySupervisedDataset")
        print("================================================")


        train_dataset = WebDatasetLazySupervisedDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            model_configs=model_configs
        )
    else:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,
                                    model_configs=model_configs,
                                    )
    data_collator_kwargs = {
            'tokenizer': tokenizer,
        }

    if hasattr(data_args, 'max_images_per_sample'):
        data_collator_kwargs['max_images_per_sample'] = data_args.max_images_per_sample



    if hasattr(data_args, 'video_max_frames'):
        data_collator_kwargs['video_max_frames'] = data_args.video_max_frames
    if hasattr(data_args, 'miv_token_len'):
        data_collator_kwargs['miv_token_len'] = data_args.miv_token_len

    # constant tokens-per-image (compile-time constant for TPU)
    tpi = model_configs.vision_tower_aux_token_len_list[0] if hasattr(model_configs, 'vision_tower_aux_token_len_list') else 256
    data_collator_kwargs['tokens_per_image'] = tpi

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# TPU Note:The TorchXLA FSDP only takes in FP32 weight. This will create an issue when you load a very large model (>30b params) on TPU in FP32. 
# TPU-V4, for example, has 100GB of memory, and a 30b model will take up at least 120GB of memory. So the solution here is to load the model in bf16.
# Then, we rewrote the FSDP sharding code to convert the bf16 weights to FP32 weights only when shard the weight. Hence, we can use minimal memory to load and shard the model on TPU.

import torch_xla
import os
XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


@torch.no_grad()
def _shard_parameters_(self, params_to_shard) -> None:
    """
    At initialization we wrap a module with full parameters and shard the
    parameters in-place. Sharding is implemented by viewing each parameter
    as a 1D Tensor and retaining only a single slice, where the slice size
    is determined by the number of data parallel workers.

    Wrapping modules with many small parameters (or with a very large data
    parallel world size) will result in many small parameter shards and slow
    performance. In this case it's better to set *``flatten_parameters``* to
    ``True``, so that all of the small parameters in the module are combined
    into a single contiguous Tensor and sharded once.

    After this initial sharding is complete, the user can initialize a
    ``torch.optim.Optimizer`` in the usual way, i.e.::

    .. code-block:: python

        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    The optimizer will see only a single slice of parameters and will thus
    allocate less memory for optimizer state, avoiding redundancy across
    data parallel workers.

    Note: this method is implemented in a different manner from
    ``fairscale.nn.FullyShardedDataParallel``. Here we delete the original
    module parameters and create new sharded parameter tensors (instead of
    making sharded tensors an attribute of the original parameters). This
    make it easier to handle things (e.g. freeing parameters) on XLA.
    """

    #print_rank0("I actually use this to shard models!")
    if len(params_to_shard) > 0:
      # When freeing the full parameters, we point their internal XLATensor to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, dtype=self.compute_dtype, device=self.xla_device)

    # get the module names of each full parameter to shard
    params_to_shard_set = set(params_to_shard)
    assert len(params_to_shard_set) == len(params_to_shard), \
        "params_to_shard should not have dups"
    full_param_infos = []
    shared_full_param_memo = {}
    shared_full_param_infos = []
    full_params = []
    for module_name, m in self.named_modules():
      for n, p in m.named_parameters(recurse=False):
        if p.dtype != torch.float32:
          #raise TypeError("only fp32 parameters are supported")
          p.data = p.data.to(torch.float32)
        if p in params_to_shard_set:
          if p in shared_full_param_memo:
            mname, shared_m, shared_n = shared_full_param_memo[p]
            shared_full_param_infos.append(
                (module_name, mname, m, n, shared_m, shared_n))
          else:
            shared_full_param_memo[p] = (module_name, m, n)
            full_param_infos.append((module_name, m, n))
            full_params.append(p)
    assert len(full_params) == len(params_to_shard_set), \
        f"there are parameters in params_to_shard not belonging to this module."
    del shared_full_param_memo
    self.full_params = full_params
    self.full_param_infos = full_param_infos
    self.shared_full_param_infos = shared_full_param_infos

    # allocate and register new sharded parameters
    self.sharded_params = []
    for idx, (module_name, m, n) in enumerate(self.full_param_infos):
        p = self.full_params[idx]
        assert not hasattr(p, "_is_sharded")

        shard_data = self._get_shard(p)

        if shard_data.device != self.xla_device:
            # cast to XLA device if not already on XLA
            shard_data = shard_data.to(self.xla_device)
        p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
        p_shard._is_sharded = True
        p_shard._orig_size = p.size()
        p_shard._orig_name = f"{module_name}.{n}"
        p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
            ".", "_FSDP_SHARD_SEPARATOR_")
        self.register_parameter(p_shard._name, p_shard)
        self.sharded_params.append(p_shard)
        if p.device != self.xla_device:
            # cast to XLA device if not already on XLA
            p = p.to(self.xla_device).requires_grad_(p.requires_grad)
            # update p in full_params since id(p) changed after the casting
            self.full_params[idx] = p
        # Free the full parameter storage (here we free its internal XLATensor) but keep the tensor itself
        # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
        if XLA_DISABLE_FUNCTIONALIZATION:
            p.data = p.new_zeros(1)  # Old behavior before Functionalization.
        elif IS_XLA_AVAILABLE:
            import torch_xla
            torch_xla._XLAC._replace_xla_tensor(p, p.new_zeros(1))
        else:
            raise RuntimeError("XLA is not available")
        p._sharded_param = p_shard  # add a handle to the sharded parameter
        p._has_full_param = False
        # deregister the full parameter tensors from their modules (so that they won't
        # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
        # only the sharded parameters should appear in the FSDP model's `parameters()`)
        assert n in m._parameters
        m._parameters.pop(n)
        object.__setattr__(m, n, p)

    # also deregister the shared parameters
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
        assert n in m._parameters
        m._parameters.pop(n)
        shared_p = getattr(shared_m, shared_n)
        object.__setattr__(m, n, shared_p)

    assert len(self.sharded_params) == len(self.full_params)

if IS_XLA_AVAILABLE:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
    XlaFullyShardedDataParallel._shard_parameters_ = _shard_parameters_




###############################################################################
## MetaMorph Change: Add W&B Recall to also record Image Autoregressive Loss ##
###############################################################################

from transformers.integrations.integration_utils import WandbCallback

def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v

            # Log the loss_language if available
            if hasattr(model, "loss_language") and model.loss_language is not None:
                logs["loss_language"] = model.loss_language
                
            # Only log image losses if they're not the "no images" default values
            # (MSE=0.0 and cosine=1.0 means no images in batch)
            has_valid_image_loss = True
            if model.vision_loss == "diffusion-loss" or model.vision_loss == "ddt-loss":
                if (hasattr(model, "loss_image_diff") and model.loss_image_diff is not None):
                # If MSE is exactly 0.0 and cosine is exactly 1.0, it's likely the "no images" case
                    if abs(model.loss_image_diff) < 1e-6:
                        has_valid_image_loss = False
                    # Only log image losses if they're valid
                if has_valid_image_loss:
                    if hasattr(model, "loss_image_diff") and model.loss_image_diff is not None:
                        logs["loss_image_diff"] = model.loss_image_diff

                # Optionally log auxiliary regression metrics alongside diffusion
                aux_enabled = (
                    getattr(model, "aux_regression_enabled", False)
                    or (hasattr(model, "config") and getattr(model.config, "aux_regression", False))
                )
                if aux_enabled and hasattr(model, "loss_image_mse") and hasattr(model, "loss_image_cos"):
                    if model.loss_image_mse is not None:
                        logs["loss_image_mse"] = model.loss_image_mse
                    if model.loss_image_cos is not None:
                        logs["loss_image_cos"] = model.loss_image_cos
                            
            else:
            # Check if we have the suspicious "no images" pattern
                if (hasattr(model, "loss_image_mse") and hasattr(model, "loss_image_cos") and
                    model.loss_image_mse is not None and model.loss_image_cos is not None):
                    # If MSE is exactly 0.0 and cosine is exactly 1.0, it's likely the "no images" case
                    if abs(model.loss_image_mse) < 1e-6 and abs(model.loss_image_cos - 1.0) < 1e-6:
                        has_valid_image_loss = False
    
                
                # Only log image losses if they're valid
                if has_valid_image_loss:
                    if hasattr(model, "loss_image_cos") and model.loss_image_cos is not None:
                        logs["loss_image_cos"] = model.loss_image_cos
                    
                    if hasattr(model, "loss_image_mse") and model.loss_image_mse is not None:
                        logs["loss_image_mse"] = model.loss_image_mse

            

            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})

######################################################################################
## End of MetaMorph Change: Add W&B Recall to also record Image Autoregressive Loss ##
######################################################################################



def check_image_tokens_simple(data_dict, images, vision_token_len, max_length):
    """Simplified version of the check_image_tokens logic for prefiltering"""
    input_ids = data_dict["input_ids"]
    num_images = len(images)
    placeholder = -200
    
    # Check 1: Placeholder count must match number of images
    ph_count = (input_ids == placeholder).sum().item()
    if ph_count != num_images:
        return False
    
    # Check 2: Full expansion must fit
    expanded_len = input_ids.numel() + (vision_token_len - 1) * num_images
    if expanded_len > max_length:
        return False
        
    return True

def should_reject_sample(sample_data, data_dict, has_image, images, vision_token_len, tokenizer):
    """
    Returns (should_reject: bool, rejection_reason: str)
    """
    
    # 1. All tokens masked (no learning signal)
    if (data_dict['labels'] != IGNORE_INDEX).sum() == 0:
        return True, "all_tokens_masked"
    
    # 2. Image-specific checks (ONLY for samples with images)
    if has_image:
        # 2a. Too many images
        if len(images) > (tokenizer.model_max_length // vision_token_len) - 1:
            return True, "too_many_images"
        
        # 2b. Length check (this is our main target!)
        if not check_image_tokens_simple(data_dict, images, vision_token_len, tokenizer.model_max_length):
            return True, "image_length_exceeded"
    
    # 3. Text-only samples: just normal tokenizer truncation (NO rejection)
    # They get truncated but are still valuable for training
    
    return False, "valid"

def run_simple_prefiltering(dataset, output_path: str):
    """
    Simple prefiltering: process all samples, reject only problematic ones
    """
    print_rank0(f"🔍 Starting simple prefiltering...")
    print_rank0(f"📊 Total samples: {len(dataset):,}")
    
    valid_count = 0
    failed_stats = {
        'image_length_exceeded': 0,    # Our main target
        'too_many_images': 0,
        'all_tokens_masked': 0,
        'tokenization_error': 0,
        'other_error': 0
    }
    
    vision_token_len = dataset.data_args.vision_tower_aux_token_len_list[0]
    tokenizer = dataset.tokenizer
    
    with open(output_path, 'w') as out_file:
        for i in range(len(dataset)):
            try:
                # Read original JSON
                with open(dataset.data_path, "rb") as f:
                    f.seek(dataset.offsets[i])
                    original_line = f.readline().decode('utf-8').strip()
                
                sample_data = json.loads(original_line)
                has_image = dataset._has_image(sample_data)
                has_video = dataset._has_video(sample_data)
                
                # Mock image loading for samples with images
                images = []
                if has_image:
                    # Assume 1 image as specified
                    images = [None]  # Don't actually load, just count
                
                # Process text (same as original)
                sources = [sample_data]
                if has_image or has_video:
                    if DEFAULT_IMAGE_TOKEN not in json.dumps(sample_data['conversations']):
                        sample_data['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sample_data['conversations'][0]['value']
                
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    dataset.data_args)
                
                data_dict = preprocess(sources, tokenizer, has_image=has_image or has_video)
                
                if isinstance(data_dict["input_ids"], list):
                    data_dict = dict(
                        input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0]
                    )
                
                # Apply rejection criteria
                should_reject, reason = should_reject_sample(
                    sample_data, data_dict, has_image, images, vision_token_len, tokenizer)
                
                if not should_reject:
                    # ✅ VALID: Write original JSON
                    out_file.write(original_line + '\n')
                    valid_count += 1
                else:
                    # ❌ REJECTED: Count statistics
                    failed_stats[reason] += 1
                    
            except Exception as e:
                failed_stats['other_error'] += 1
                continue
            
            # Progress
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / len(dataset) * 100
                print_rank0(f"⏳ Progress: {i+1:,}/{len(dataset):,} ({progress:.1f}%) - Valid: {valid_count:,}")
    
    # Statistics
    total_failed = sum(failed_stats.values())
    print_rank0(f"✅ Prefiltering completed!")
    print_rank0(f"📊 Statistics:")
    print_rank0(f"   • Valid samples: {valid_count:,} ({valid_count/len(dataset)*100:.1f}%)")
    print_rank0(f"   • Rejected: {total_failed:,} ({total_failed/len(dataset)*100:.1f}%)")
    for reason, count in failed_stats.items():
        if count > 0:
            print_rank0(f"     - {reason}: {count:,}")

def train(INDEX, attn_implementation=None):

    # ###############
    # ## For Debug ##
    # ###############
    # import torch_xla.core.xla_model as xm


    # # 1. Define and create the cache directory
    # #    (ensure all processes can write here, or use rank-specific paths if needed)
    # cache_dir = "./xla_cache_rank_{}".format(INDEX) # Example: rank-specific path
    # # Or a shared path: cache_dir = "./xla_cache"
    # os.makedirs(cache_dir, exist_ok=True)

    # # 2. Enable the persistent cache *early*
    # #    Set readonly=True for ranks > 0 if using a shared path might be safer,
    # #    but typically each process manages its cache or XLA handles concurrency.
    # #    Start simple with potentially separate dirs or a shared one.
    # xr.initialize_cache(cache_dir, readonly=False)




    import wandb
    import torch_xla.core.xla_model as xm
    if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        logger.info("Run with torchxla spmd...")
        if torch.distributed.get_rank() == 0:
            if os.getenv('WANDB_API_KEY', None) is not None:
                wandb.login(key=os.getenv('WANDB_API_KEY'))
                # Early wandb init to capture all command line output
                wandb.init(
                    project=os.getenv('WANDB_PROJECT', 'huggingface'),
                    name=os.getenv('WANDB_NAME', ''),
                )
        else:
            os.environ["WANDB_MODE"] = "disabled" # ! NOTE: disable wandb for non-master node

    elif os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_MP":
        logger.info("Run with torchxla mp...")
        if os.getenv('WANDB_API_KEY', None) is not None and xm.get_ordinal() == 0:
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            # Early wandb init to capture all command line output
            wandb.init(
                project=os.getenv('WANDB_PROJECT', 'huggingface'),
                name=os.getenv('WANDB_NAME', ''),
            )

    global local_rank
    
    log_rank0(f"Training on index {INDEX}. Local rank: {local_rank}")


    rank       = dist.get_rank()        # 0 … 63   ← the host you're on
    world_size = dist.get_world_size()  # 64       ← total TPU‑VMs


    logger.warning(f"Rank: {rank}, World size: {world_size}")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # verify that the train_batch_size is set correctly
    if training_args.batch_size is not None:
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            world_size = xm.xrt_world_size()

            if training_args.per_device_train_batch_size is None:
                raise ValueError("If train_batch_size is set, per_device_train_batch_size must be set")

            if training_args.batch_size != training_args.per_device_train_batch_size * world_size:
                raise ValueError(f"train_batch_size ({training_args.train_batch_size}) must equal per_device_train_batch_size ({training_args.per_device_train_batch_size}) * world_size ({world_size})")

            logger.warning(f"per_device_train_batch_size is correctly set to {training_args.per_device_train_batch_size} with world_size {world_size} to match train_batch_size {training_args.batch_size}")
            logger.warning(f"train_batch_size is {training_args.train_batch_size}")

    
    # TPU Note, the original LLaMA RMSNorm implementation has a bug here, the dtype conversion is not correct. It is ok in GPU but kills TPU training.
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = (self.weight * hidden_states).to(input_dtype)
        return output

    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = forward
    transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward

    # def new_forward_conv(self, input):

        
    #     if self.bias is None:
    #         return self._conv_forward(input, self.weight, self.bias)

    #     # print("whyyyyy, input is", input)
    #     return self._conv_forward(input, self.weight, self.bias.to(input.dtype))

    # nn.Conv2d.forward = new_forward_conv

    # def new_forward_linear(self, input):
    #     if self.bias is None:
    #         return F.linear(input, self.weight, self.bias)
    #     return F.linear(input, self.weight, self.bias.to(input.dtype)).to(input.dtype)

    # nn.Linear.forward = new_forward_linear

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    else:
        log_rank0(f"Loading model in full precision")

    use_cohere = False
    data_args.si_token_len = model_args.si_token_len
    data_args.miv_token_len = model_args.miv_token_len

    if model_args.vision_tower_aux_list is not None:
        # copy image_token_len and image_position to model_args
        # data_args.image_token_len = model_args.image_token_len
        # model_args.image_position = data_args.image_position

        # Assuming model_args.model_name_or_path is a string that includes the model size
        model_name = model_args.model_name_or_path

        # Regular expression to find the number of parameters in the model's name (assuming a convention like 'ModelName-30b')
        match = re.search(r'(\d+)b', model_name)
        num_parameters_billion = float(match.group(1)) if match else 0

        # Determine if bfloat16 should be used based on the model's size
        use_bfloat16 = training_args.bf16 or num_parameters_billion > 30

        if "yi" in model_args.model_name_or_path.lower():
            use_bfloat16 = True

        if "qwen" in model_name.lower():
            logger.warning(f"Vision tower, loading ScaleRAEQwenForCausalLM: {model_args.model_name_or_path}")
            debug_print("bnb_model_from_pretrained_args is", bnb_model_from_pretrained_args)
            # exit()
            # replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer

            if (
                hasattr(training_args, 'fsdp_config') and
                'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
            ):
                logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with Qwen2DecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
                if model_args.vision_loss == "diffusion-loss" or model_args.vision_loss == "ddt-loss":
                    if model_args.diffusion_split_per_token>1:
                        training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer", "LightningDiTBlock"]
                    else:
                        training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer", "ResBlock"]
                        # training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer"]
                    if model_args.dit_cls == "xattnDiT":
                        training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer", "LuminaNextDiTBlock"]
                else:
                    training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["Qwen2DecoderLayer"]

            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                # Add any other necessary config loading args like trust_remote_code=True if needed
            )

            config.vision_loss = model_args.vision_loss
            config.vision_loss_mode = model_args.vision_loss_mode
            config.vision_coef = model_args.vision_coef
            config.diffusion_split_per_token = model_args.diffusion_split_per_token
            # Dual-loss (diffusion + regression) options
            config.aux_regression = getattr(model_args, 'aux_regression', False)
            config.aux_regression_coef = getattr(model_args, 'aux_regression_coef', 1.0)
            # K tiling per sample
            config.diffusion_timesteps_per_sample = getattr(model_args, 'diffusion_timesteps_per_sample', 1)

            if model_args.diffusion_model_hidden_size_II !=0:
                assert model_args.diffusion_model_depth_II !=0 and model_args.diffusion_model_heads_II
                model_args.diffusion_model_hidden_size = [model_args.diffusion_model_hidden_size, model_args.diffusion_model_hidden_size_II]
                model_args.diffusion_model_depth = [model_args.diffusion_model_depth, model_args.diffusion_model_depth_II]
                model_args.diffusion_model_heads = [model_args.diffusion_model_heads, model_args.diffusion_model_heads_II]  


            config.diffusion_model_hidden_size = model_args.diffusion_model_hidden_size
            config.diffusion_model_channels = model_args.diffusion_model_channels
            config.diffusion_model_depth = model_args.diffusion_model_depth
            config.diffusion_model_heads = model_args.diffusion_model_heads
            config.diffusion_model_z_channels = model_args.diffusion_model_z_channels
            config.ddt_encoder_depth = model_args.ddt_encoder_depth
            config.diffusion_class_dropout_prob = model_args.diffusion_class_dropout_prob
            # Select DiT backbone for diffusion head (DiT/DDT/xattnDiT)
            config.dit_cls = getattr(model_args, 'dit_cls', 'DiT')

            # pass through optional base dim for diffusion
            if hasattr(model_args, 'diffusion_base_dim'):
                config.diffusion_base_dim = model_args.diffusion_base_dim
            if hasattr(model_args, 'generation_alignment_tower'):
                config.patch_size = 2

            # NEW: plumb through normalization stats path
            config.diffusion_norm_stats_path = model_args.diffusion_norm_stats_path


            model = ScaleRAEQwenForCausalLM.from_pretrained(
                    model_name,
                    config=config, 
                    # cache_dir=training_args.cache_dir,
                    # do_sample=True,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
            )
            transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm.forward = forward

            debug_print("This is immediately after loading the model")
            # model.diff_head.model._print_weight()

            # exit()


        else:
            raise ValueError(f"Unsupported model type in model_name: {model_name}. Scale-RAE only supports Qwen2-based models.")
    else:
        logger.warning(f"No vision tower, loading pure language model: {model_args.model_name_or_path}")
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.generation_config.do_sample = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    log_rank0("Model loaded.")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        log_rank0("Using gradient checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        log_rank0("Adding LoRA adapters...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        print_rank0("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    log_rank0("Configuring tokenizer...")
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "llama_v3":
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    elif model_args.version == 'qwen_2':
        # follow config (https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer_config.json) and instructions (https://github.com/QwenLM/Qwen2.5/issues/486)
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            logger.warning(f"Conversation version {model_args.version} not found. Using default `vicuna_v1`")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    log_rank0(f"Default conversation version: {conversation_lib.default_conversation.version}")
    print_rank0("Then it is", conversation_lib.default_conversation)

    if use_cohere:
        tokenizer.pad_token_id = 0
        print_rank0("tokenizer id is", tokenizer.pad_token_id)
    # print_rank0("tokenizer is", tokenizer)

    if model_args.vision_tower_aux_list is not None:
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(model_args.vision_tower_aux_token_len_list)
        # model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        model.load_vision_head(model_args=model_args)

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter



        
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower

        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()
        
        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(dtype=torch.bfloat16 if training_args.bf16 else None, device=training_args.device)
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            tune_modules = ['mm_projector']
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    print_rank0('tuning {}'.format(name))
                    param.requires_grad = True

        
        # --- NEW: tune_adapter_and_vision_head ---
        model.config.tune_adapter_and_vision_head = training_args.tune_adapter_and_vision_head = model_args.tune_adapter_and_vision_head

        model.config.pretrain_adapter_and_vision_head = training_args.pretrain_adapter_and_vision_head = model_args.pretrain_adapter_and_vision_head


        
        if model_args.tune_adapter_and_vision_head:
            model.requires_grad_(False)
            tune_modules = ['mm_projector', 'vision_head', 'aux_vision_head', 'diff_head', "latent_queries", 'embed_tokens']
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    print_rank0('tuning {}'.format(name))
                    param.requires_grad = True
            # Vision head (regression or diffusion)
            for name, param in model.named_parameters():
                if ("vision_head" in name) or ("aux_vision_head" in name) or ("diff_head" in name) or ("latent_queries" in name):
                    print_rank0('tuning vision head: {}'.format(name))
                    param.requires_grad = True
                    
        # --- NEW: tune_vision_head (head-only; no adapter/tokenizer changes) ---
        model.config.tune_vision_head = training_args.tune_vision_head = model_args.tune_vision_head
        if model_args.tune_vision_head:
            model.requires_grad_(False)
            tune_modules = ['diff_head', 'diff_head_projector', 'aux_vision_head', 'latent_queries']
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    print_rank0('tuning {}'.format(name))
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        if training_args.bits in [4, 8]:
            log_rank0(f"Initializing vision modules in {training_args.bits}bit")
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        # model.config.image_token_len = data_args.image_token_len = model_args.image_token_len
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        model.config.diff_head_lr = training_args.diff_head_lr
        
        model.config.generation_alignment_tower = model_args.generation_alignment_tower

        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = data_args.vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        # model.config.image_token_len = data_args.image_token_len

        model.config.si_token_len = data_args.si_token_len = model_args.si_token_len
        model.config.miv_token_len = data_args.miv_token_len = model_args.miv_token_len

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if model_args.generation_alignment_tower is not None:
        from scale_rae.model.multimodal_encoder.builder import build_vision_tower
        print_rank0(f"Loading generation alignment tower: {model_args.generation_alignment_tower}, type: {type(model_args)}")

        alignment_tower_args = copy.deepcopy(model_args)
        alignment_tower_args.mm_vision_tower = model_args.generation_alignment_tower
        alignment_tower_args.unfreeze_mm_vision_tower = False # freeze VAE alignment tower

        # Store like SigLIP: plain Python list to avoid FSDP/module registration
        _vae_tower = build_vision_tower(alignment_tower_args)
        _vae_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float32, device=training_args.device)
        _vae_tower.requires_grad_(False)

        model.generation_alignment_tower_list = [_vae_tower]

        print_rank0(f"Setting up generation alignment processor")
        data_args.image_processor_gen = _vae_tower.image_processor

        # Ensure no registered submodule remains on the model under old attribute
        if hasattr(model, 'generation_alignment_tower'):
            setattr(model, 'generation_alignment_tower', None)

    else:
        model.generation_alignment_tower_list = []
        data_args.image_processor_gen = None

    if training_args.bits in [4, 8]:
        log_rank0(f"Initializing model in {training_args.bits}bit")
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    log_rank0("Configuring data module...")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model_configs=model.config,
                                              )

    # 🚀 HIJACK: Dataset prefiltering mode
    if training_args.prefilter_dataset:
        if training_args.prefilter_output_path is None:
            # Auto-generate output path  
            base_name = os.path.splitext(data_args.data_path)[0]
            training_args.prefilter_output_path = f"{base_name}_filtered_ctx{training_args.model_max_length}.jsonl"
        
        print_rank0("🚀 PREFILTER MODE: Starting dataset prefiltering instead of training...")
        print_rank0(f"📂 Input: {data_args.data_path}")
        print_rank0(f"📂 Output: {training_args.prefilter_output_path}")
        
        run_simple_prefiltering(
            dataset=data_module['train_dataset'],
            output_path=training_args.prefilter_output_path
        )
        
        print_rank0("🏁 Prefiltering complete. Exiting without training.")
        return  # Exit without creating trainer

    if training_args.bf16:
        model = model.to(dtype=torch.float32)

    ########################################################################################
    ########################################################################################
    # log_rank0(f"Model: \n{model}")

    if isinstance(model.model.vision_tower_aux_list, list):
        log_rank0(f"Vision towers: \n{model.model.vision_tower_aux_list}")
        log_rank0(f"Seems vision encoder is not training.")
    
    if training_args.load_weights:
        log_rank0(f"Loading weights from {training_args.load_weights}")
        msg = model.load_state_dict(load_file(training_args.load_weights), strict=False)
        log_rank0(f"Loading weights: {msg}")

    log_rank0("Configuring trainer...")

    verbose = [["name", "shape", "trainable?"]]
    for name, param in model.named_parameters():
        verbose.append([name, param.shape, param.requires_grad])
    # log_rank0(tabulate(verbose, headers="firstrow", tablefmt='pipe'))


    # Check if image start/end tokens are special tokens
    if model_args.mm_use_im_start_end:
        print_rank0(f"Using image start/end tokens: {DEFAULT_IM_START_TOKEN}, {DEFAULT_IM_END_TOKEN}")
        # Check if these are in the tokenizer's special tokens
        print_rank0(f"Special tokens: {tokenizer.special_tokens_map}")
        
        # Get token IDs if they exist
        if DEFAULT_IM_START_TOKEN in tokenizer.get_vocab():
            im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            im_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            print_rank0(f"Image start token ID: {im_start_id}, Image end token ID: {im_end_id}")
        
            # @MetaMoprh changes: Remember image start and end 
            model.im_start_id, model.im_end_id = im_start_id, im_end_id


        else:
            print_rank0(f"Image tokens not found in vocab. They may be added during initialization.")
    
    
    debug_print("Before trainer")
    
    # model.diff_head.model._print_weight()
    print("Model architecture:", model)

    trainer = ScaleRAETrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    # callbacks=[CustomWandbCallback()],
                    **data_module)
    
    debug_print("After trainer")
    
    # model.diff_head.model._print_weight()
    
    trainer.is_fsdp_enabled = True

    resume_from_checkpoint=training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # else:
    #     trainer.train()

    # After training is complete, we force one last sharded save.
    # We call the internal sharded save method directly.
    # The arguments `model`, `trial`, and `metrics` are standard for the Trainer's internal API.
    # We can pass the trainer's own model and None for the others.
    log_rank0("Forcing a final sharded save before consolidated save...")
    trainer._save_checkpoint(model=trainer.model, trial=None, metrics=None)
    trainer.control = trainer.callback_handler.on_save(trainer.args, trainer.state, trainer.control)  
    log_rank0("Final sharded save complete.")

    log_rank0(f"Training finished: {training_args.output_dir}")
    
    trainer.save_state()

    model.config.use_cache = True

    log_rank0("Saving model...")
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
