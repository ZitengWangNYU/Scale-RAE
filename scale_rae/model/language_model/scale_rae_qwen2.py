#    Copyright 2024 Hao Zhang
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

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
# import torch_xla.amp as amp # Can use torch.autocast as well

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

from ..scale_rae_arch import ScaleRAEMetaModel, ScaleRAEMetaForCausalLM, apply_custom_kernel

from scale_rae.utils import IS_XLA_AVAILABLE

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm # <-- Import XLA model



def check_params_for_nan(model: nn.Module, model_name: str = "Model"):
    """
    Iterates through model parameters and checks if any contain NaN values.

    Args:
        model (nn.Module): The PyTorch model instance to check.
        model_name (str): Optional name for the model for clearer print statements.

    Returns:
        bool: True if any NaNs were found in parameters, False otherwise.
    """
    found_nan = False
    print(f"\n--- Checking parameters for NaNs in {model_name} ---")
    for name, param in model.named_parameters():
        if not param.requires_grad: # Skip parameters that don't require gradients (e.g., buffers)
            continue
        if torch.isnan(param.data).any():
            print(f"!!! NaN detected in parameter: {name}, shape: {param.shape}")
            found_nan = True
        # Optional: Check for Infs too
        # if torch.isinf(param.data).any():
        #     print(f"!!! Inf detected in parameter: {name}, shape: {param.shape}")
        #     found_nan = True # Treat Inf as problematic too

    if not found_nan:
        print(f"--- No NaNs found in the parameters of {model_name}. ---")
    else:
         print(f"--- NaN detected in parameters of {model_name}! ---")
    return found_nan


def ensure_float32(module: nn.Module):
    """
    Recursively iterates through a module's parameters and buffers,
    ensuring they are float32.

    Args:
        module (nn.Module): The PyTorch module to convert.
    """
    xm.mark_step()  # Mark the step for XLA

    print(f"Attempting to convert module '{module.__class__.__name__}' and its children to float32...")
    conversion_count = 0
    buffer_conversion_count = 0

    for name, param in module.named_parameters(recurse=True):

        print("I am at least surverying", name, param.dtype)

        if param.dtype != torch.float32:
            try:
                print("converting:", name)
                param.data = param.data.to(torch.float32)
                # Ensure requires_grad status is preserved if it was True
                if param.requires_grad:
                     param.requires_grad_(True)
                conversion_count += 1
            except Exception as e:
                 print(f"  Failed to convert parameter '{name}': {e}")



    print(f"Completed conversion. Converted {conversion_count} parameters and {buffer_conversion_count} buffers.")

    # --- Verification Step (Optional but recommended) ---
    print("\nVerifying dtypes after conversion...")
    all_float32 = True
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != torch.float32:
            print(f"  Verification FAILED: Parameter '{name}' is still {param.dtype}")
            all_float32 = False
    for name, buf in module.named_buffers(recurse=True):
        if torch.is_floating_point(buf) and buf.dtype != torch.float32:
            print(f"  Verification FAILED: Buffer '{name}' is still {buf.dtype}")
            all_float32 = False

    if all_float32:
        print("  Verification PASSED: All parameters and float buffers are now float32.")
    else:
         print("  Verification FAILED: Not all parameters/buffers were converted to float32.")

    xm.mark_step()  # Mark the step for XLA


def test_rectified_flow_projector(rf_proj):
    """
    Test the RectifiedFlowProjector class.
    """
    # Create a dummy input tensor
    z = torch.randn(6, rf_proj.z_channels)
    x = torch.randn(6, rf_proj.diffusion_tokens, rf_proj.diffusion_channels)

    print("z and y dypes are:", z.dtype, x.dtype)

    # Forward pass
    output = rf_proj(z, x)
    print(f"--- [FORWARD] output shape: {output.shape} --")
    
    
    # loss compute 
    loss = rf_proj.training_loss(z, x)
    print(f"--- [LOSS] loss: {loss} --")
    
    # inference
    
    x_end = rf_proj.infer(z)
    print(f"--- [INFER] x_end shape: {x_end.shape} --")




class ScaleRAEQwenConfig(Qwen2Config):
    model_type = "cambrian_qwen"
    #@Peter: Hardcode diffusion loss for now, need to be changed later
    vision_loss = "regression-loss"
    vision_loss_mode = "causal"
    vision_tower_aux_token_len_list = [256]  # Default vision token length


class ScaleRAEQwenModel(ScaleRAEMetaModel, Qwen2Model):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config: Qwen2Config):
        if IS_XLA_AVAILABLE:
            config._attn_implementation = "eager"
        super(ScaleRAEQwenModel, self).__init__(config)


    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:





        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
                if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
                    # ! NOTE@shusheng: this is a hack to speed up the training
                    # ! NOTE@shusheng: we use torch_xla's flash attention which does not require mask
                    attention_mask = None
                else:
                    # 4d mask is passed through the layers
                    attention_mask = _prepare_4d_causal_attention_mask(
                        attention_mask,
                        (batch_size, seq_length),
                        inputs_embeds,
                        past_key_values_length,
                        sliding_window=self.config.sliding_window,
                    )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def compute_feature_loss(predictions, targets, valid_positions, loss_type='l2'):
    """
    Compute per-token feature prediction loss, normalized by feature dimension.
    
    Args:
        predictions: Tensor of shape [bs, seq_len, dim]
        targets: Tensor of shape [bs, seq_len, dim]
        valid_positions: Tensor of shape [bs, seq_len] - binary mask (1 for valid tokens)
        loss_type: One of 'l1', 'l2', or 'smooth_l1'
    
    Returns:
        Mean loss per valid token, normalized by feature dimension
    """
    # Expand mask to feature dimension
    mask_expanded = valid_positions.unsqueeze(-1).expand_as(predictions)
    
    # Get feature dimension for normalization
    feature_dim = predictions.size(-1)
    
    # Apply loss function
    if loss_type == 'l1':
        # L1 loss (Mean Absolute Error)
        diff = torch.abs(predictions - targets)
    elif loss_type == 'smooth_l1':
        # Smooth L1 loss (Huber loss)
        diff = torch.nn.functional.smooth_l1_loss(
            predictions, targets, reduction='none')
    else:  # default to 'l2'
        # L2 loss (Mean Squared Error)
        diff = (predictions - targets)**2
    
    # Mask the differences
    masked_diff = diff * mask_expanded
    
    # Sum across feature dimension and divide by feature dim to get average per dimension
    # This gives us per-token loss normalized by feature dimension
    per_token_loss = masked_diff.sum(dim=-1) / feature_dim  # [bs, seq_len] 
    
    # Get mean of only valid tokens in a TPU-compatible way
    epsilon = 1e-8
    total_loss = per_token_loss.sum()
    num_valid = valid_positions.sum() + epsilon
    mean_loss = total_loss / num_valid
    
    return mean_loss

from ..diffusion_loss.diffloss import create_rf_projector

class ScaleRAEQwenForCausalLM(Qwen2ForCausalLM, ScaleRAEMetaForCausalLM):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        config.rope_scaling = None

        self.model = ScaleRAEQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

        self.vision_loss = getattr(config, 'vision_loss', 'diffusion-loss')
        self.vision_loss_mode = getattr(config, 'vision_loss_mode', 'causal')
        self.vision_coef = getattr(config, 'vision_coef', 1.0)
        self.vision_tower_aux_token_len_list = getattr(config, 'vision_tower_aux_token_len_list', [256])  # Default vision token length
        self.diffusion_model_channels = getattr(config, 'diffusion_model_channels', 1152)
        # num_image_tokens: Use diffusion_split_per_token for DiT-based models (what DiT expects)
        # For WebMAE and similar models where si_token_len == diffusion_split_per_token, this also works
        # Default to 256 for backward compatibility
        self.num_image_tokens = getattr(config, 'diffusion_split_per_token', 256)
        self.debug = False
        if self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
            if self.vision_loss_mode == 'causal':
                self.diff_head_config =  {
                    "diffusion_tokens": 1, # default = 1, we actually hardcode causal diffusion to 1 token
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels": config.hidden_size, # Qwen-2.5 7B 3584
                    "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                    "model_depth": config.diffusion_model_depth, # default = 12
                    "model_heads": config.diffusion_model_heads,
                    "guidance_scale": 2.0,
                }
                
                self.diff_head_config["use_mlp"] = True
                # NEW: optional normalization stats path
                if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                    self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path

            elif self.vision_loss_mode == 'ar-ddt':
                print("Using AR-DDT diffusion-loss for image feature prediction, split_per_token, model_hidden_size, model_depth and diffusion_model_z_channels:", config.diffusion_split_per_token, config.diffusion_model_hidden_size, config.diffusion_model_depth, config.diffusion_model_z_channels)
                
                # Check if we should use DDT architecture
                use_ddt = (self.vision_loss == "ddt-loss")
                print(f"AR-DDT mode with vision_loss='{self.vision_loss}', use_DDT={use_ddt}")
                
                self.diff_head_config = {
                    "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 256
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # Qwen-2.5 7B 3584
                    "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                    "model_depth": config.diffusion_model_depth, # default = 12
                    "model_heads": config.diffusion_model_heads,
                    "guidance_scale": 1.0,
                    "use_mlp": False,
                    "use_DDT": use_ddt,  # Enable DDT when vision_loss == "ddt-loss"
                }
                if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                    self.diff_head_config["base_dim"] = config.diffusion_base_dim
                # NEW: optional normalization stats path
                if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                    self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path
                
                # Add DDT-specific parameters when using DDT
                if use_ddt:
                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")
                        
                    self.diff_head_config.update({
                        "DDT_encoder_depth": config.ddt_encoder_depth,
                        "class_dropout_prob": cls_prob
                    })

                if config.diffusion_model_z_channels != 0:
                    self.diff_head_projector = nn.Linear(config.hidden_size, config.diffusion_model_z_channels)
                    self.use_diff_head_projector = True
                else:
                    self.use_diff_head_projector = False



            elif self.vision_loss_mode == 'query' or self.vision_loss_mode == 'query-block':

                if self.vision_loss == "ddt-loss":

                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")

                    self.diff_head_config =  {
                        "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 1
                        "diffusion_channels": self.diffusion_model_channels,
                        "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # QEwen-2.5 7B 3584
                        "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                        "model_depth": config.diffusion_model_depth, # default = 12
                        "model_heads": config.diffusion_model_heads,
                        "guidance_scale": 1.0,
                        "use_mlp": False,
                        "class_dropout_prob": cls_prob
                    }
                    if type(config.diffusion_model_hidden_size) != list:
                        # This means the classic DDT
                        self.diff_head_config["use_DDT"] = True
                        self.diff_head_config["DDT_encoder_depth"] = config.ddt_encoder_depth
                                        # optional: pass base dimension for diffusion scaling if defined
                    if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                        self.diff_head_config["base_dim"] = config.diffusion_base_dim
                    # NEW: optional normalization stats path
                    if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                        self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path


                
                
                else:
                    
                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")


                    self.diff_head_config =  {
                        "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 1
                        "diffusion_channels": self.diffusion_model_channels,
                        "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # QEwen-2.5 7B 3584
                        "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                        "model_depth": config.diffusion_model_depth, # default = 12
                        "model_heads": config.diffusion_model_heads,
                        "guidance_scale": 1.0,
                        "cond_silu": True, # assume the run is before 0711
                        "class_dropout_prob": cls_prob
                    }
                    if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                        self.diff_head_config["base_dim"] = config.diffusion_base_dim
                    # NEW: optional normalization stats path
                    if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                        self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path

                
                if config.diffusion_model_z_channels != 0:
                    self.diff_head_projector = nn.Linear(config.hidden_size, config.diffusion_model_z_channels)
                    self.use_diff_head_projector = True
                else:
                    self.use_diff_head_projector = False

                self.diff_head_config["use_mlp"] = False

            
            # Ensure diffusion backbone selection is propagated
            self.diff_head_config["dit_cls"] = getattr(config, 'dit_cls', 'DiT')

            # # Add use_mlp=True if diffusion_tokens is 1RF
            # if self.diff_head_config["split_per_token"] == 1:
            #     print("Calling from MLP")
            #     self.diff_head_config["use_mlp"] = True
            # else:
            #     print("Calling from DiT")
            #     self.diff_head_config["use_mlp"] = False


            self.diff_head = create_rf_projector(self.diff_head_config)

            # self.conditioning_preprocessor = nn.Sequential(
            #     nn.LayerNorm(config.hidden_size),
            #     # nn.Linear(config.hidden_size, config.hidden_size), # Or project to diff_head.z_channels if different
            #     # nn.LayerNorm(config.hidden_size),
            #     # You could add an activation here too if desired, e.g., nn.GELU()
            # )

            self.set_to_fp32 = False

            # Optional: auxiliary regression head trained alongside diffusion
            self.aux_regression_enabled = getattr(config, 'aux_regression', False)
            self.aux_regression_coef = getattr(config, 'aux_regression_coef', 1.0)
            if self.aux_regression_enabled:
                self.aux_vision_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, 1152),
                )




        elif self.vision_loss == 'regression-loss':
            print("Using regression-loss for image feature prediction")
            self.vision_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1152),
            )

        # exit()

        self.post_init()

    def load_vision_head(self, model_args):
        pretrain_adapter_and_vision_head = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        print(f"pretrain_adapter_and_vision_head is: {pretrain_adapter_and_vision_head}")

        if pretrain_adapter_and_vision_head is not None:
            print(f"[DEBUG] Loading adapter and vision head weights from: {pretrain_adapter_and_vision_head}")
            adapter_vision_weights = torch.load(pretrain_adapter_and_vision_head, map_location='cpu')
            print(f"[DEBUG] Keys in loaded checkpoint: {list(adapter_vision_weights.keys())}")
            
            # Extract from 'model' key if it exists
            if 'model' in adapter_vision_weights:
                model_weights = adapter_vision_weights['model']
                print(f"[DEBUG] Keys in model: {list(model_weights.keys())}")
            else:
                model_weights = adapter_vision_weights

            def get_w(weights, keyword):
              return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword+'.' in k}
            
            
            # Load vision_head if present
            if hasattr(self, 'vision_head') and any('vision_head.' in k for k in model_weights.keys()):
                print("[DEBUG] Loading vision_head weights.")
                self.vision_head.load_state_dict(get_w(model_weights, 'vision_head'), strict=False)
            else:
                print("[DEBUG] No vision_head weights found in the checkpoint, skipping loading.")
            
            # Load diff_head if present
            if hasattr(self, 'diff_head') and any('diff_head.' in k for k in model_weights.keys()):
                print("[DEBUG] Loading diff_head weights.")
                self.diff_head.load_state_dict(get_w(model_weights, 'diff_head'), strict=False)
            else:
                print("[DEBUG] No diff_head weights found in the checkpoint, skipping loading.")

    # <<< --- ADD THIS OVERRIDE --- >>>
    def _init_weights(self, module):
        """
        Override the base _init_weights. Initialize base model parts as default,
        but SKIP re-initialization for our custom diff_head.
        """
        # Check if the module belongs to the diff_head
        is_in_diff_head = False
        if hasattr(self, 'diff_head'):
            for name, child_module in self.diff_head.named_modules():
                if module is child_module:
                    is_in_diff_head = True
                    break

        if is_in_diff_head:
            # If it's part of diff_head, do nothing - let its own init stand.
            pass
        else:
            # If it's part of the base model, call the parent's _init_weights
            # (which contains the standard Qwen2 initialization logic)
            super()._init_weights(module) # Calls Qwen2ForCausalLM's _init_weights


    def set_diff_fp32(self):
        if not self.set_to_fp32:
            ensure_float32(self.diff_head.model)
            self.set_to_fp32 = True


    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        vision_token_indices: Optional[torch.Tensor] = None,
        decoding: Optional[bool] = False,
        answer_img_mask: Optional[torch.Tensor] = None,
        reverse_vti: Optional[torch.Tensor] = None,
        answer_token_mask: Optional[torch.Tensor] = None,
        guidance_level: Optional[float] = 1.0,
        images_gen: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        selected_features, input_embed_mask, attention_bias, extra_mm = None, None, None, None

        if inputs_embeds is None:
            (
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
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                vision_token_indices=vision_token_indices,
                answer_img_mask=answer_img_mask,
                reverse_vti=reverse_vti,
                answer_token_mask=answer_token_mask,
                images_gen=images_gen,
            )
        
        # Store attention bias for training patch to access
        if attention_bias is not None:
            # Store on all attention layers so the training patch can access it
            for layer in self.get_model().layers:
                layer.self_attn._current_attention_bias = attention_bias
        else:
            # Clear attention bias when not needed
            for layer in self.get_model().layers:
                if hasattr(layer.self_attn, '_current_attention_bias'):
                    layer.self_attn._current_attention_bias = None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            #self.model.gradient_checkpointing = False
                
            from torch_xla.utils.checkpoint import checkpoint
            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # training
        if IS_XLA_AVAILABLE:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else: # inference
            if hasattr(self, "vision_tower_aux_feature_list"):
                raise NotImplementedError("vision_tower_aux_feature_list should not be set in inference mode")
            else:
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = outputs[0]


        logits = self.lm_head(hidden_states)
        
        
        if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            import torch_xla.distributed.spmd as xs
            
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(logits, xs.get_global_mesh(), ("fsdp", None, "mp"))


            
        logits = logits.float()


        if decoding:
            vision_loss_mode = self.vision_loss_mode
            use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
            # Currently, this code assumes regression loss
            generated_token_length = 1 if not use_query_mode else self.num_image_tokens
            if self.vision_loss == 'regression-loss':
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1)  # (B, hidden_dim) for next toekn, (B, L, hidden_dim) for multiple tokens 

                print(f"[DEBUG] Hidden states shape before regression loss: {hidden_states.shape}, pred_z shape: {pred_z.shape}")
                
                pred_z = self.vision_head(pred_z)

                prediction = self.get_model().mm_projector(pred_z)

                pred_patches = pred_z
                prediction_target = answer_token_mask if answer_token_mask is not None else None

                
                valid_mask_patch = torch.ones(prediction.shape[0], prediction.shape[1], dtype=torch.int, device=prediction.device)

                feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                feature_loss = compute_feature_loss(
                    pred_patches,                # (B,T,F)
                    prediction_target,           # (B,T,F)
                    valid_mask_patch,            # (B,T)
                    loss_type=feature_loss_type
                )

                eps = 1e-8

                # Cosine loss (patch-level)
                norm_pred = torch.nn.functional.normalize(pred_patches + eps, dim=-1)
                norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                cosine_sim = (norm_pred * norm_tgt).sum(-1)  # (B,T)
                masked_cos = cosine_sim * valid_mask_patch
                avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                cosine_loss = 1.0 - avg_cosine

                print("[DEBUG] Feature loss:", feature_loss.item(), "Cosine loss:", cosine_loss.item())
                # exit()


                hidden_states[:, -generated_token_length:, :] = prediction
                
                print(f"[DEBUG] Hidden states shape after regression loss: {hidden_states.shape}, pred_z shape: {pred_z.shape}, prediction shape: {prediction.shape}")
            elif self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1) 
                hidden_pred_z = pred_z.clone().detach()
                 # (B, hidden_dim) for next toekn, (B, L, hidden_dim) for multiple tokens
                
                # For multi-GPU: move input to diff_head's device instead of moving diff_head
                # This avoids issues with accelerate hooks that keep modules on their assigned devices
                diff_head_device = next(self.diff_head.parameters()).device
                pred_z = pred_z.to(diff_head_device)
                
                if self.use_diff_head_projector:
                    self.diff_head_projector = self.diff_head_projector.to(diff_head_device)
                    pred_z = self.diff_head_projector(pred_z)

                pred_z = self.diff_head.infer(pred_z, guidance_level=guidance_level)

                # Move pred_z back to mm_projector's device for projection
                try:
                    mm_proj = self.get_model().mm_projector
                    mm_proj_device = next(mm_proj.parameters()).device
                    pred_z = pred_z.to(mm_proj_device)
                    prediction = mm_proj(pred_z)
                except:
                    prediction = hidden_pred_z.clone().detach()

                # Move prediction back to hidden_states device
                prediction = prediction.to(hidden_states.device)
                hidden_states[:, -generated_token_length:, :] = prediction
                
            
            else:
                raise NotImplementedError(f"Decoding mode not implemented for vision_loss type: {self.vision_loss}")

            

        if decoding:
            return CausalLMOutputWithPast(
                loss=pred_z,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )






        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)


            self.loss_language = loss



            vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')

            # ------------------------------------------------------------------
            # QUERY-MODE  (answer-image tokens are latent queries)
            # ------------------------------------------------------------------
            if vision_loss_mode_cfg == "query" or vision_loss_mode_cfg == "query-block":

                if self.vision_loss == 'regression-loss':
                    # Expecting extra_mm = (image_features, reverse_vti, answer_img_mask)
                    img_feats_raw, reverse_vti, answer_img_mask, prediction_target = extra_mm

                    B, T, feature_dim = img_feats_raw.shape
                    M = answer_img_mask.size(1)
                    tokens_per_image = T // M

                    hidden_dim = hidden_states.size(-1)

                    # 1. Hidden states aligned to sequence indices (pad dummy at t=0)
                    # hs_pred = hidden_states[:, :-1]  # (B, L-1, hidden)
                    # pad_zero_hs = torch.zeros(B, 1, hidden_dim, dtype=hs_pred.dtype, device=hs_pred.device)
                    # hs_full = torch.cat([pad_zero_hs, hs_pred], dim=1)  # (B, L, hidden)
                    hs_full  = hidden_states  # (B, L, hidden)

                    # 2. Build zeros placeholder for left chunk (B,T,hidden)
                    zeros_left_hs = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)

                    # 3. Gather hidden states using kernel-ready reverse_vti
                    patch_hs = apply_custom_kernel(
                        zeros_left_hs,  # length T (image side)
                        hs_full,        # length Lmax text side
                        reverse_vti     # kernel-ready indices (B,T)
                    )  # (B,T,hidden)

                    # # --------------------------- DEBUG SHAPES ---------------------------
                    # dbg_shape("img_feats_raw", img_feats_raw)
                    # dbg_shape("reverse_vti", reverse_vti)
                    # dbg_shape("answer_img_mask", answer_img_mask)
                    # dbg_shape("prediction_target", prediction_target)
                    # dbg_shape("patch_hs", patch_hs)

                    # 5. Predict features from gathered hidden states
                    pred_patches = self.vision_head(patch_hs)  # (B,T,feat)

                    # dbg_shape("pred_patches", pred_patches)

                    # ---- Patch-level mask identical to non-query branch ----
                    valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)  # (B,M,P)
                    valid_mask_patch = valid_mask_patch.reshape(B, T).int()

                    eps = 1e-8

                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    feature_loss = compute_feature_loss(
                        pred_patches,                # (B,T,F)
                        prediction_target,           # (B,T,F)
                        valid_mask_patch,            # (B,T)
                        loss_type=feature_loss_type
                    )

                    # Cosine loss (patch-level)
                    norm_pred = torch.nn.functional.normalize(pred_patches + eps, dim=-1)
                    norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                    cosine_sim = (norm_pred * norm_tgt).sum(-1)  # (B,T)
                    masked_cos = cosine_sim * valid_mask_patch
                    avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                    cosine_loss = 1.0 - avg_cosine

                    img_loss = feature_loss + cosine_loss
                    loss = loss + img_loss * self.vision_coef

                    self.loss_image_mse = feature_loss
                    self.loss_image_cos = cosine_loss

                elif self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
                    # --------------------------------------------------------------
                    # Diffusion loss (query-mode)
                    # --------------------------------------------------------------

                    # Unpack multimodal outputs
                    img_feats_raw, reverse_vti, answer_img_mask, prediction_target = extra_mm

                    B, T, feature_dim = prediction_target.shape
                    M = answer_img_mask.size(1)
                    tokens_per_image = T // M

                    hidden_dim = hidden_states.size(-1)

                    # Gather hidden states corresponding to each image patch via reverse_vti
                    hs_full = hidden_states  # (B, L, hidden)
                    zeros_left_hs = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)
                    patch_hs = apply_custom_kernel(zeros_left_hs, hs_full, reverse_vti)  # (B, T, hidden)


                    # 1. Reshape inputs for training_loss: (B, T, dim) -> (B*M, tokens_per_image, dim)
                    # patch_hs: (B, T, hidden_dim) -> (B, M, tokens_per_image, hidden_dim) -> (B*M, tokens_per_image, hidden_dim)
                    patch_hs_reshaped = patch_hs.view(B, M, tokens_per_image, hidden_dim).view(B*M, tokens_per_image, hidden_dim)
                    if self.use_diff_head_projector:
                        # Project to diffusion model z_channels if needed
                        patch_hs_reshaped = self.diff_head_projector(patch_hs_reshaped)

                    
                    # prediction_target: (B, T, feature_dim) -> (B, M, tokens_per_image, feature_dim) -> (B*M, tokens_per_image, feature_dim)
                    prediction_target_reshaped = prediction_target.view(B, M, tokens_per_image, feature_dim).view(B*M, tokens_per_image, feature_dim)

                    # Compute diffusion loss per image, with optional K-tiling of timesteps per sample.
                    # We reuse internal timestep sampling by passing t=None.
                    K = int(getattr(self.config, 'diffusion_timesteps_per_sample', 1) or 1)
                    if K > 1:
                        # Tile along the image batch dimension
                        z_tiled = torch.repeat_interleave(patch_hs_reshaped, repeats=K, dim=0)              # (B*M*K, P, hidden)
                        x_tiled = torch.repeat_interleave(prediction_target_reshaped, repeats=K, dim=0)     # (B*M*K, P, feat)
                        loss_vec = self.diff_head.training_loss(z=z_tiled, x=x_tiled)                        # (B*M*K,)
                        diffusion_loss_per_image = loss_vec.view(B * M, K).mean(dim=1)                       # (B*M,)
                    else:
                        diffusion_loss_per_image = self.diff_head.training_loss(
                            z=patch_hs_reshaped, x=prediction_target_reshaped
                        )  # (B*M,)
                    
                    # 2. Reshape back to (B, M) for image-level masking
                    diffusion_loss_per_image = diffusion_loss_per_image.view(B, M)
                    
                    # Apply image-level masking using answer_img_mask (B, M)
                    masked_loss_per_image = diffusion_loss_per_image * answer_img_mask.float()


                    eps = 1e-8
                    mean_diffusion_loss = masked_loss_per_image.sum() / (answer_img_mask.sum() + eps)

                    # Aggregate into total loss
                    loss = loss + mean_diffusion_loss * self.vision_coef

                    self.loss_image_diff = mean_diffusion_loss


                    # --------------------------------------------------------------
                    # Optional auxiliary regression loss (query-mode)
                    # --------------------------------------------------------------
                    if getattr(self, 'aux_regression_enabled', False):
                        # Reuse gathered hidden states and targets
                        pred_patches_aux = self.aux_vision_head(patch_hs)  # (B,T,feat)
                        valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)
                        valid_mask_patch = valid_mask_patch.reshape(B, T).int()

                        feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                        aux_feature_loss = compute_feature_loss(
                            pred_patches_aux,
                            prediction_target,
                            valid_mask_patch,
                            loss_type=feature_loss_type
                        )
                        # optional cosine term mirroring regression branch
                        eps = 1e-8
                        norm_pred = torch.nn.functional.normalize(pred_patches_aux + eps, dim=-1)
                        norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                        cosine_sim = (norm_pred * norm_tgt).sum(-1)
                        masked_cos = cosine_sim * valid_mask_patch
                        avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                        aux_cosine_loss = 1.0 - avg_cosine

                        aux_img_loss = aux_feature_loss + aux_cosine_loss
                        loss = loss + self.aux_regression_coef * aux_img_loss
                        self.loss_image_aux_reg = aux_img_loss
                        # Log aux components under standard names for callbacks
                        self.loss_image_mse = aux_feature_loss
                        self.loss_image_cos = aux_cosine_loss


                else:
                    raise ValueError(f"Unsupported vision_loss '{self.vision_loss}' in query mode")

            # ------------------------------------------------------------------
            # AR-DDT MODE (noisy patches fed to LLM for enhanced conditioning)
            # ------------------------------------------------------------------
            elif vision_loss_mode_cfg == "ar-ddt":
                # Expecting extra_mm = (x_t, t_ar, answer_img_mask, pred_image_features, reverse_vti)
                x_t, t_ar, answer_img_mask, pred_image_features, reverse_vti = extra_mm

                # Shapes
                #   pred_image_features : (B, T_total, feat_dim)
                #   x_t                 : (B, T_total, feat_dim)
                #   t_ar               : (B_img,) where B_img = B * M (image-level batch)

                B, T_total, feature_dim = pred_image_features.shape
                M = answer_img_mask.size(1)                               # images per sample
                tokens_per_image = T_total // M                           # patches per image

                hidden_dim = hidden_states.size(-1)

                # ------------------------------------------------------
                # 1) Gather hidden states for *each patch* via reverse_vti
                # ------------------------------------------------------
                hs_full = hidden_states                                   # (B, L, hidden)
                zeros_left_hs = torch.zeros(B, T_total, hidden_dim,
                                            dtype=hs_full.dtype, device=hs_full.device)
                patch_hs = apply_custom_kernel(zeros_left_hs, hs_full, reverse_vti)  # (B, T_total, hidden)

                # ------------------------------------------------------
                # 2)  Optional projection to z_channels
                # ------------------------------------------------------
                if self.use_diff_head_projector:
                    patch_hs = self.diff_head_projector(patch_hs)         # (B, T_total, z_channels)

                # ------------------------------------------------------
                # 3)  Reshape to image-level batches that the diffusion
                #     head expects:  (B*M, tokens_per_image, dim)
                # ------------------------------------------------------
                patch_hs_img = patch_hs.view(B, M, tokens_per_image, -1).view(B * M, tokens_per_image, -1)
                x_clean_img  = pred_image_features.view(B, M, tokens_per_image, feature_dim).view(B * M, tokens_per_image, feature_dim)
                x_t_img      = x_t.view(B, M, tokens_per_image, feature_dim).view(B * M, tokens_per_image, feature_dim)

                # t_ar is already on image-level (B*M,)
                t_img = t_ar.view(-1)

                # ------------------------------------------------------
                # 4)  Diffusion loss with external t and x_t
                # ------------------------------------------------------
                diffusion_loss_per_img = self.diff_head.training_loss(
                    z=patch_hs_img,
                    x=x_clean_img,
                    t=t_img,
                    x_t=x_t_img,
                )  # (B*M,)

                # ------------------------------------------------------
                # 5)  Mask & aggregate over answer images only
                # ------------------------------------------------------
                diffusion_loss_per_img = diffusion_loss_per_img.view(B, M)  # (B, M)

                # answer_img_mask: 1 for answer images, 0 for context images
                masked_loss_per_img = diffusion_loss_per_img * answer_img_mask.float()

                eps = 1e-8
                mean_diffusion_loss = masked_loss_per_img.sum() / (answer_img_mask.sum() + eps)

                # ------------------------------------------------------
                # 6)  Add to total loss & log
                # ------------------------------------------------------
                loss = loss + mean_diffusion_loss * self.vision_coef
                self.loss_image_diff = mean_diffusion_loss
 

                # Optional auxiliary regression loss (AR-DDT): use clean x as target
                if getattr(self, 'aux_regression_enabled', False):
                    pred_patches_aux = self.aux_vision_head(patch_hs)  # (B, T_total, feat)
                    # Mask to answer images' tokens
                    valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)
                    valid_mask_patch = valid_mask_patch.reshape(B, T_total).int()
                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    aux_feature_loss = compute_feature_loss(
                        pred_patches_aux,
                        x_clean_img.view(B, T_total, feature_dim),
                        valid_mask_patch,
                        loss_type=feature_loss_type
                    )
                    eps = 1e-8
                    norm_pred = torch.nn.functional.normalize(pred_patches_aux + eps, dim=-1)
                    norm_tgt  = torch.nn.functional.normalize(x_clean_img.view(B, T_total, feature_dim) + eps, dim=-1)
                    cosine_sim = (norm_pred * norm_tgt).sum(-1)
                    masked_cos = cosine_sim * valid_mask_patch
                    avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                    aux_cosine_loss = 1.0 - avg_cosine
                    aux_img_loss = aux_feature_loss + aux_cosine_loss
                    loss = loss + self.aux_regression_coef * aux_img_loss
                    self.loss_image_aux_reg = aux_img_loss
                    # Log aux components under standard names
                    self.loss_image_mse = aux_feature_loss
                    self.loss_image_cos = aux_cosine_loss

            # ------------------------------------------------------------------
            # NON-QUERY modes   →  keep existing logic unchanged
            # ------------------------------------------------------------------
            else:
                hidden_states_for_prediction = hidden_states[:, :-1]  # States 0 to SeqLen-2
                
                if self.vision_loss == 'diffusion-loss' and self.debug == False:
                    # (existing diffusion branch stays verbatim)

                # Pad the hidden states to match the sequence length of targets/mask
                    bs, seq_len, _ = selected_features.shape  # Use seq_len from selected_features
                    hidden_dim = hidden_states_for_prediction.shape[2]
                    padding_tensor = torch.zeros(hidden_states_for_prediction.shape[0], 1, hidden_states_for_prediction.shape[2],
                                            dtype=hidden_states_for_prediction.dtype, device=hidden_states_for_prediction.device)
                    hidden_states_padded = torch.cat((padding_tensor, hidden_states_for_prediction), dim=1)

                    valid_positions = input_embed_mask.int()
                    reshaped_hidden_states = hidden_states_padded.view(bs * seq_len, hidden_dim)
                    reshaped_hidden_states = reshaped_hidden_states.to(dtype=selected_features.dtype)
                    feature_dim = selected_features.shape[2]
                    diffusion_tokens = self.diff_head_config['diffusion_tokens']
                    reshaped_target_features = selected_features.view(bs * seq_len, 1, feature_dim)
                    diffusion_loss_flat = self.diff_head.training_loss(
                        z=reshaped_hidden_states,
                        x=reshaped_target_features
                        )
                    diffusion_loss_flat = diffusion_loss_flat.squeeze()
                    diffusion_loss_reshaped = diffusion_loss_flat.view(bs, seq_len)
                    masked_loss = diffusion_loss_reshaped * valid_positions.float()
                    epsilon = 1e-8
                    mean_diffusion_loss = masked_loss.sum() / (valid_positions.sum() + epsilon)
                    self.loss_image_diff = mean_diffusion_loss
                    loss = loss + mean_diffusion_loss * self.vision_coef

                else:
                    # existing regression branch (causal / block modes)
                    predicted_features_raw = self.vision_head(hidden_states_for_prediction)
                    padding_tensor = torch.zeros(predicted_features_raw.shape[0], 1, predicted_features_raw.shape[2],
                                                dtype=predicted_features_raw.dtype, device=predicted_features_raw.device)
                    all_predicted_features = torch.cat((padding_tensor, predicted_features_raw), dim=1)
                    valid_positions = input_embed_mask.int()
                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    feature_loss = compute_feature_loss(all_predicted_features, selected_features, valid_positions, loss_type=feature_loss_type)
                    epsilon = 1e-8
                    norm_pred = torch.nn.functional.normalize(all_predicted_features + epsilon, dim=-1)
                    norm_target = torch.nn.functional.normalize(selected_features + epsilon, dim=-1)
                    cosine_sim = (norm_pred * norm_target).sum(dim=-1)
                    masked_cosine_sim = cosine_sim * valid_positions
                    num_valid = valid_positions.sum() + epsilon
                    avg_cosine_sim = masked_cosine_sim.sum() / num_valid
                    cosine_loss = 1.0 - avg_cosine_sim
                    img_loss = feature_loss + cosine_loss
                    loss = loss + img_loss * self.vision_coef
                    self.loss_image_mse = feature_loss
                    self.loss_image_cos = cosine_loss


        

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


    def greedy_decode(self, position_ids, attention_mask, inputs_embeds, start_image_token_id, end_image_token_id, eos_token_id, do_sample=None, temperature=None, top_p=None, num_beams=None, max_new_tokens=1024, use_cache=None, output_image=False, extra_mm=None, guidance_level=None):
        
        # Convert eos_token_id to list if it's not already a list
        if eos_token_id is not None and not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]
        past_key_values = None
        in_image_mode = False
        generated_ids_list = []
        total_image_tokens = 0
        total_output_tokens = 0
        
        use_cache = False
        
        image_embeds_list = []


        # Initialize attention_mask if it's None
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)

        num_image_tokens = self.num_image_tokens 
        vision_loss_mode = self.vision_loss_mode
        use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
        while True:
           
            attention_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

            outputs = self.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                decoding=in_image_mode, 
                answer_token_mask=extra_mm, # Debugging note, we pass extra_mm here
                guidance_level=guidance_level,
            )

            image_embed = outputs.loss
            if in_image_mode:
                generated_token_length = 1 if not use_query_mode else num_image_tokens
            else:
                generated_token_length = 1
            next_token_logits = outputs.logits[:, -generated_token_length:, :].squeeze(1) # backward compatibility
            next_embed = outputs.hidden_states[:, -generated_token_length:, :].squeeze(1)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            next_token_embed = self.model.embed_tokens(next_token)
           

            if (not in_image_mode) and next_token.item() == start_image_token_id:
                in_image_mode = True
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                if use_query_mode:
                    latent_query = self.get_model().latent_queries.unsqueeze(0)
                    expanded_latent_query = latent_query.expand(inputs_embeds.size(0), -1, -1)
                    inputs_embeds = torch.cat((inputs_embeds, expanded_latent_query), dim=1)


            elif (in_image_mode) and (total_image_tokens<num_image_tokens):

                total_image_tokens += generated_token_length # directly generate all tokens at once if use queries

                #image_embeds_list.append(image_embed)
                image_embeds_list.extend(image_embed)

                if use_query_mode:
                    inputs_embeds[:, -self.num_image_tokens:, :] = next_embed
                else:
                    inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
                    
                # inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

                if total_image_tokens==num_image_tokens:
                    in_image_mode = False
                    if use_query_mode:
                        break


            elif next_token.item() == end_image_token_id:
                in_image_mode = False
                total_image_tokens = 0
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)

            
            else:
                # Append token embeddings
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                generated_ids_list.extend(next_token.squeeze(0).tolist())

            total_output_tokens += 1

            if next_token.numel() == 1 and next_token.item() in eos_token_id: # only judge eos for single token pred
                break

            if total_output_tokens > max_new_tokens:
                break

            past_key_values = outputs.past_key_values
          
        if image_embeds_list:
            image_embeds_tensor = torch.cat(image_embeds_list, dim=0)
        else:
            image_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]


         # Perform random cropping and compute cosine similarity
        if output_image:
            return output, image_embeds_tensor

        return output, None


    def greedy_decode_with_logits(self, position_ids, attention_mask, inputs_embeds, start_image_token_id, end_image_token_id, eos_token_id, do_sample=None, temperature=None, top_p=None, num_beams=None, max_new_tokens=1024, use_cache=None, output_image=False, extra_mm=None, guidance_level=None):
        


        # Convert eos_token_id to list if it's not already a list
        if eos_token_id is not None and not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]
        past_key_values = None
        in_image_mode = False
        generated_ids_list = []
        total_image_tokens = 0
        total_output_tokens = 0
        
        use_cache = False
        
        image_embeds_list = []
        logits_list = []  # Store logits for each generated token
        conf_scores_list = []  # Store confidence scores for each generated token


        # Initialize attention_mask if it's None
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)

        num_image_tokens = self.num_image_tokens 
        vision_loss_mode = self.vision_loss_mode
        use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
        while True:
           
            attention_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

            outputs = self.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                decoding=in_image_mode, 
                answer_token_mask=extra_mm, # Debugging note, we pass extra_mm here
                guidance_level=guidance_level,
            )

            image_embed = outputs.loss
            if in_image_mode:
                generated_token_length = 1 if not use_query_mode else num_image_tokens
            else:
                generated_token_length = 1
            next_token_logits = outputs.logits[:, -generated_token_length:, :].squeeze(1) # backward compatibility
            next_embed = outputs.hidden_states[:, -generated_token_length:, :].squeeze(1)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)            # Extract logit value for the selected token only
            selected_token_logit = torch.gather(next_token_logits, -1, next_token.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token_conf_scores = -torch.log(self.config.vocab_size * next_token_probs).mean()

            next_token_embed = self.model.embed_tokens(next_token)
           

            if (not in_image_mode) and next_token.item() == start_image_token_id:
                print("Enter image mode")
                in_image_mode = True
                #generated_ids_list.append(next_token.item())
                print("I am entering image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                print('next token is:', next_token, next_token.squeeze(0).tolist())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (start_image_token)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                if use_query_mode:
                    latent_query = self.get_model().latent_queries.unsqueeze(0)
                    expanded_latent_query = latent_query.expand(inputs_embeds.size(0), -1, -1)
                    print("[Appending Query] inputs_embeds shape before appending:", inputs_embeds.shape, 'query shape:', expanded_latent_query.shape)
                    inputs_embeds = torch.cat((inputs_embeds, expanded_latent_query), dim=1)


            elif (in_image_mode) and (total_image_tokens<num_image_tokens):

                total_image_tokens += generated_token_length # directly generate all tokens at once if use queries

                #image_embeds_list.append(image_embed)
                print("I am in image mode!!!, this image sequence has number of tokens:", total_image_tokens, 'image embed shape:', image_embed.shape, 'next embed shape:', next_embed.shape)
                image_embeds_list.extend(image_embed)
                print('image_embeds_list shape:', len(image_embeds_list))

                if use_query_mode:
                    inputs_embeds[:, -self.num_image_tokens:, :] = next_embed
                else:
                    inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
                    
                # inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

                if total_image_tokens==num_image_tokens:
                    in_image_mode = False
                    if use_query_mode:
                        print("I am leaving image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                        break


            elif next_token.item() == end_image_token_id:
                print("I am leaving image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                in_image_mode = False
                total_image_tokens = 0
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (end_image_token)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)

            
            else:
                # Append token embeddings
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (regular text tokens)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))

            total_output_tokens += 1

            if next_token.numel() == 1 and next_token.item() in eos_token_id: # only judge eos for single token pred
                break

            if total_output_tokens > max_new_tokens:
                break

            past_key_values = outputs.past_key_values
          
        if image_embeds_list:
            image_embeds_tensor = torch.cat(image_embeds_list, dim=0)
        else:
            image_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]
        
        # Convert logits_list to tuple of tensors (similar to HuggingFace format)
        logits = tuple(logits_list) if logits_list else None
        conf_scores = tuple(conf_scores_list) if conf_scores_list else None

         # Perform random cropping and compute cosine similarity
        if output_image:
            # print('final image embed shape:', image_embeds_tensor.shape)
            # print("output, image embeds are:", image_embeds_tensor)
            return output, image_embeds_tensor, logits, conf_scores

        return output, None, logits, conf_scores


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        use_customize_greedy: Optional[bool] = False,
        return_scores: Optional[bool] = False,
        start_image_token_id = None,
        end_image_token_id = None,
        eos_token_id=None,
        guidance_level=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")


        extra_mm = kwargs.pop("extra_mm", None) # extra_mm is only passed when debug_with_vision is True
        if images is not None or image_embeds is not None:
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


            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=images,
                image_embeds=image_embeds,
            )
            # self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            # self.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
            # self.final_vision_feature_size = final_vision_feature_size
            # self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)


        if use_customize_greedy:
            if return_scores:
                return self.greedy_decode_with_logits(position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    start_image_token_id=start_image_token_id,
                    end_image_token_id=end_image_token_id,
                    eos_token_id=eos_token_id,
                    extra_mm=extra_mm,
                    guidance_level=guidance_level,
                    **kwargs
                )
            else:
                return self.greedy_decode(position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    start_image_token_id=start_image_token_id,
                    end_image_token_id=end_image_token_id,
                    eos_token_id=eos_token_id,
                    extra_mm=extra_mm,
                    guidance_level=guidance_level,
                    **kwargs
                )
        else:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images

        return inputs


def dbg_shape(name: str, tensor):  # noqa: ANN001
    try:
        print(f"[QDBG] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    except AttributeError:
        print(f"[QDBG] {name}: {tensor}")


AutoConfig.register("cambrian_qwen", ScaleRAEQwenConfig)
AutoModelForCausalLM.register(ScaleRAEQwenConfig, ScaleRAEQwenForCausalLM)