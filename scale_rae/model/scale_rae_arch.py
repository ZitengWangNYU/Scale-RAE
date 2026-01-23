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


from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ezcolorlog import root_logger as logger

from .multimodal_encoder.builder import build_vision_tower_aux_list
from .multimodal_projector.builder import build_vision_projector

from scale_rae.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from scale_rae.utils import IS_XLA_AVAILABLE

from .diffusion_loss.diffloss import create_rf_projector
if IS_XLA_AVAILABLE:
    import torch_xla.distributed.spmd as xs

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
class CustomKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_embeds, img_embeds, token_indices):
        ctx.full_input_shape = input_embeds.shape
        ctx.full_img_shape = img_embeds.shape
        ctx.dtype = input_embeds.dtype
        ctx.device = input_embeds.device

        
        sharded_input_embeds = xs.enable_manual_sharding(input_embeds, ("fsdp", None, None)).global_tensor
        sharded_img_embeds = xs.enable_manual_sharding(img_embeds, ("fsdp", None, None)).global_tensor
        sharded_token_indices = xs.enable_manual_sharding(token_indices, ("fsdp", None)).global_tensor

        sharded_embeds = torch.cat([sharded_input_embeds, sharded_img_embeds], dim=1)
        sharded_embeds = torch.gather(sharded_embeds, 1, sharded_token_indices.unsqueeze(-1).expand(-1, -1, sharded_embeds.size(-1)))
        output_embeds = xs.disable_manual_sharding(sharded_embeds, ("fsdp", None, None), input_embeds.shape, mesh=xs.get_global_mesh()).global_tensor

        ctx.save_for_backward(token_indices)
        return output_embeds

    @staticmethod
    def backward(ctx, grad_output):

        bs = ctx.full_input_shape[0]
        input_seqlen = ctx.full_input_shape[1]
        img_seqlen = ctx.full_img_shape[1]
        dim = ctx.full_input_shape[2]
        token_indices, = ctx.saved_tensors
        
        sharded_token_indices = xs.enable_manual_sharding(token_indices, ("fsdp", None)).global_tensor
        sharded_grad_output = xs.enable_manual_sharding(grad_output, ("fsdp", None, None)).global_tensor
        lbs = sharded_grad_output.shape[0]

        sharded_embeds_grad = torch.zeros(
            lbs, input_seqlen + img_seqlen, dim,
            dtype=ctx.dtype, device=sharded_token_indices.device)
        sharded_embeds_grad = sharded_embeds_grad.scatter_add(1, sharded_token_indices.unsqueeze(-1).expand(-1, -1, dim), sharded_grad_output)

        full_grad_shape = (bs, input_seqlen + img_seqlen, dim)
        full_grad = xs.disable_manual_sharding(sharded_embeds_grad, ("fsdp", None, None), full_grad_shape, mesh=xs.get_global_mesh()).global_tensor

        return full_grad[:, :input_seqlen].clone(), full_grad[:, input_seqlen:].clone(), None

def apply_custom_kernel(input_embeds, img_embeds, token_indices):
    return CustomKernel.apply(input_embeds, img_embeds, token_indices)


class CustomScatterKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tgt_embeds, src_embeds, indices):
        """Scatter src_embeds (length-K) into tgt_embeds (length-T) along dim=1.
        indices is (B,K) giving target column per src row.
        All tensors already have identical dtype/device.
        """
        ctx.save_for_backward(indices)
        ctx.tgt_shape = tgt_embeds.shape
        ctx.src_shape = src_embeds.shape

        sh_tgt = xs.enable_manual_sharding(tgt_embeds, ("fsdp", None, None)).global_tensor
        sh_src = xs.enable_manual_sharding(src_embeds, ("fsdp", None, None)).global_tensor
        sh_idx = xs.enable_manual_sharding(indices, ("fsdp", None)).global_tensor

        sh_tgt.scatter_(1, sh_idx.unsqueeze(-1).expand_as(sh_src), sh_src)

        out = xs.disable_manual_sharding(sh_tgt, ("fsdp", None, None), ctx.tgt_shape, mesh=xs.get_global_mesh()).global_tensor
        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices, = ctx.saved_tensors
        sh_grad = xs.enable_manual_sharding(grad_out, ("fsdp", None, None)).global_tensor
        sh_idx  = xs.enable_manual_sharding(indices, ("fsdp", None)).global_tensor

        exp_idx = sh_idx.unsqueeze(-1).expand(-1, -1, sh_grad.size(-1))
        sh_tgt_grad = sh_grad.clone().scatter_(1, exp_idx, 0.)
        sh_src_grad = sh_grad.gather(1, exp_idx)

        tgt_grad = xs.disable_manual_sharding(sh_tgt_grad, ("fsdp", None, None), ctx.tgt_shape, mesh=xs.get_global_mesh()).global_tensor
        src_grad = xs.disable_manual_sharding(sh_src_grad, ("fsdp", None, None), ctx.src_shape, mesh=xs.get_global_mesh()).global_tensor
        return tgt_grad, src_grad, None

def apply_custom_scatter_kernel(tgt_embeds, src_embeds, indices):
    return CustomScatterKernel.apply(tgt_embeds, src_embeds, indices)


class ScaleRAEMetaModel:

    def __init__(self, config):
        super(ScaleRAEMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):

            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if projector_type == 'sva':
                raise NotImplementedError
            else:
                self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)
                config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in self.vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(config)
                # self.image_newline = nn.Parameter(
                #         torch.empty(config.hidden_size, dtype=self.dtype)
                #     )
        
        # Initialize latent queries (used in query / block vision-loss modes)
        vision_loss_mode_cfg = getattr(config, 'vision_loss_mode', 'causal')
        if vision_loss_mode_cfg in ['query', 'block', 'half-query', 'query-block']:
            vision_token_len = getattr(config, 'vision_tower_aux_token_len_list', [256])[0]
            embed_std = 1.0 / torch.sqrt(torch.tensor(config.hidden_size, dtype=torch.float32))
            self.latent_queries = nn.Parameter(torch.randn(vision_token_len, config.hidden_size) * embed_std)
        else:
            self.latent_queries = None

    def get_vision_tower_aux_list(self):
        vision_tower_aux_list = getattr(self, 'vision_tower_aux_list', None)
        return vision_tower_aux_list

    def initialize_vision_modules(self, model_args, fsdp=None):
        # vision_tower = model_args.vision_tower
        # num_query_group = model_args.num_query_group
        # query_num_list = model_args.query_num_list
        vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        # image_token_len = model_args.image_token_len
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_adapter_and_vision_head = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        connector_only = model_args.connector_only
        # connector_depth = model_args.connector_depth

        # self.config.mm_vision_tower = vision_tower
        # self.config.image_token_len = image_token_len
        # self.config.num_query_group = num_query_group
        # self.config.query_num_list = query_num_list
        # assert num_query_group == len(query_num_list)
        # self.config.connector_depth = connector_depth
        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only = connector_only

        if self.get_vision_tower_aux_list() is None:
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            if model_args.unfreeze_mm_vision_tower:
                self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
            else:
                # self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
                # self.vision_tower_aux_list.requires_grad_(False)
                self.vision_tower_aux_list = vision_tower_aux_list
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.vision_hidden_size = vision_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:

            if self.config.mm_projector_type == 'sva':
                raise NotImplementedError
            else:
                self.config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(self.config)
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                # self.image_newline = nn.Parameter(
                #     torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                # )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword+'.' in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict=True)

            if self.config.mm_projector_type == 'sva':
                raise NotImplementedError
            # self.image_newline.data = mm_projector_weights['model.image_newline']

       # --- NEW: Load weights for adapter and vision head if specified ---
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
            
            # Load mm_projector if present
            if hasattr(self, 'mm_projector') and any('mm_projector.' in k for k in model_weights.keys()):
                print("[DEBUG] Loading mm_projector weights.")
                self.mm_projector.load_state_dict(get_w(model_weights, 'mm_projector'), strict=False)
            else:
                print("[DEBUG] No mm_projector weights found in the checkpoint, skipping loading.")

            
            
            # --------------------------------------------------
            # Load latent_queries (trained query embeddings) if present
            # --------------------------------------------------
            if (
                hasattr(self, 'latent_queries') and self.latent_queries is not None
                and any('latent_queries' in k for k in model_weights.keys())
            ):
                lq_key = next(k for k in model_weights.keys() if 'latent_queries' in k)
                pretrained_lq = model_weights[lq_key]
                if pretrained_lq.shape == self.latent_queries.data.shape:
                    print("[DEBUG] Loading latent_queries into model.")
                    with torch.no_grad():
                        self.latent_queries.data.copy_(pretrained_lq)
                else:
                    print(
                        f"[WARN] latent_queries shape mismatch: checkpoint {pretrained_lq.shape} vs model {self.latent_queries.data.shape}. Skipping load."
                    )
            else:
                print("[DEBUG] No latent_queries found in checkpoint or model config does not use them.")





def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :]=0
            mask[:, -padding:, :]=0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding]=0
            mask[:, :, -padding:]=0
        return mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def build_block_attention_bias(
    input_ids: torch.Tensor,
    input_embed_mask: torch.Tensor,
    im_start_id: int,
    num_heads: int,
    vision_loss_mode: str = "causal",
) -> torch.Tensor:
    """Create an additive bias that *unmasks* bidirectional attention **only**
    between image tokens, while keeping the standard causal masking for every
    other pair.

    Strategy
    --------
    Flash-attention is still invoked with ``causal=True``.  That already adds
    ``-inf`` to all *future* positions (j > i) for every token.

    To re-enable future attention *within* the image block we therefore add a
    large positive value (not +inf to avoid numerical issues) for every pair
    *(i, j)* where **both** tokens are image tokens **in the same contiguous block**.
    Adding a large positive value effectively cancels the existing ‑inf so the
    softmax sees a large positive instead of ‑inf.  All other entries receive
    *zero* bias so the default causal behaviour is untouched.

    Returns
    -------
    attention_bias:  bfloat16 tensor of shape ``[B, H, L, L]`` ready to pass to
    the `flash_attention` kernel (``ab`` argument).
    """

    B, L = input_ids.shape
    device = input_ids.device

    # 1. Identify contiguous blocks of image tokens.
    # An image block is denoted by a unique ID starting from a `im_start_id`.
    is_start_token = (input_ids == im_start_id)
    block_ids = torch.cumsum(is_start_token.int(), dim=1)

    # We only care about blocks for image tokens. Text tokens will have block_id 0.
    is_image_token = input_embed_mask.bool()
    image_block_ids = block_ids * is_image_token.int()

    # 2. Create a mask for pairs of tokens in the same image block.
    # `same_block` will be True if both tokens are in the same non-zero block.
    same_block = (image_block_ids.unsqueeze(-1) == image_block_ids.unsqueeze(-2))
    # Rule out pairs where at least one is a text token (or padding).
    same_block.masked_fill_(image_block_ids.unsqueeze(-1) == 0, False)

    # 3. Create a mask for the upper triangle (future tokens, j > i).
    future_mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

    # 4. The final mask for applying bias: pairs that are in the same block AND are future tokens.
    bias_mask = same_block & future_mask

    # 5. Start with zeros (no bias).
    bias = torch.zeros((B, L, L), dtype=torch.bfloat16, device=device)

    # Match whatever constant the kernel uses for its causal "−∞" entries.
    try:
        from torch_xla.experimental.custom_kernel import FlashAttention  # type: ignore
        large_positive_bias: float = -float(FlashAttention.DEFAULT_MASK_VALUE)
    except Exception:  # noqa: BLE001
        # Fallback for CPU/CI runs where torch-xla is absent.
        large_positive_bias = 0.7 * float(torch.finfo(torch.float32).max)

    # 6. Apply large positive value only to the future pairs within the same image block.
    bias.masked_fill_(bias_mask, large_positive_bias)

    # 7. Expand to all attention heads.
    bias = bias.unsqueeze(1).expand(B, num_heads, L, L).contiguous()

    return bias



class ScaleRAEMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    # def get_vision_tower(self):
    #     return self.get_model().get_vision_tower()

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def rearrange_vision_tower_features_train(self, vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list, query_side_len):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature, vision_tower_aux_attention_masks in zip(vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list):
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature.view(bs, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0,2).flatten(1,2)

            vision_tower_aux_attention_masks_rearranged = vision_tower_aux_attention_masks.view(bs*query_side_len*query_side_len, reduce_factor*reduce_factor)

            vision_tower_aux_feature_rearranged_list.append(vision_tower_aux_feature_rearranged)
            vision_tower_aux_attention_masks_rearranged_list.append(vision_tower_aux_attention_masks_rearranged)
        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def rearrange_vision_tower_features_inference(self, vision_tower_aux_feature_list, query_side_len, image_sizes, unpad=False):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature in vision_tower_aux_feature_list:
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)

            vision_tower_aux_feature_rearranged = []
            vision_tower_aux_attention_masks_rearranged = []
            for batch_i in range(bs):
                image_size = image_sizes[batch_i]
                cur_vision_tower_aux_feature = vision_tower_aux_feature[batch_i]

                cur_vision_tower_aux_attention_masks_rearranged = torch.ones((1, aux_height, aux_width), dtype=torch.bool, device=cur_vision_tower_aux_feature.device)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5).contiguous()
                if unpad:
                    cur_vision_tower_aux_feature_rearranged = unpad_image(cur_vision_tower_aux_feature_rearranged, image_size)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.flatten(0,2).flatten(1,2) # query_side_len*query_side_len X reduce_factor*reduce_factor X C

                cur_vision_tower_aux_attention_masks_rearranged = unmask_attention_mask(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor).permute(0, 1, 3, 2, 4).contiguous()
                if unpad:
                    cur_vision_tower_aux_attention_masks_rearranged = unpad_image(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.flatten(0,2).flatten(1,2)

                cur_vision_tower_aux_attention_masks_rearranged[cur_vision_tower_aux_attention_masks_rearranged.sum(-1)==0] = True

                vision_tower_aux_feature_rearranged.append(cur_vision_tower_aux_feature_rearranged)
                vision_tower_aux_attention_masks_rearranged.append(cur_vision_tower_aux_attention_masks_rearranged)

            vision_tower_aux_feature_rearranged = torch.cat(vision_tower_aux_feature_rearranged, 0)
            vision_tower_aux_attention_masks_rearranged = torch.cat(vision_tower_aux_attention_masks_rearranged, 0)


            vision_tower_aux_feature_rearranged_list.append(vision_tower_aux_feature_rearranged)
            vision_tower_aux_attention_masks_rearranged_list.append(vision_tower_aux_attention_masks_rearranged)

        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def encode_images(self, image_aux_list):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        image_aux_features_list = []
        for image_aux, vision_tower_aux in zip(image_aux_list, vision_tower_aux_list):
            if len(image_aux.shape) == 3:
                image_aux = image_aux.unsqueeze(0)

            image_aux_features = vision_tower_aux(image_aux)
            image_aux_features_list.append(image_aux_features)

        return image_aux_features_list



    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images=None, vision_token_indices=None,
        answer_img_mask: Optional[torch.Tensor] = None,
        reverse_vti: Optional[torch.Tensor] = None,
        answer_token_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        images_gen: Optional[torch.Tensor] = None,
    ):

        if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            import torch_xla.distributed.spmd as xs

        # vision_tower = self.get_vision_tower()
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()


        if vision_tower_aux_list is None or input_ids.shape[1] == 1 or (images is None and image_embeds is None):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None

        # import ipdb; ipdb.set_trace(context=10)
        if image_embeds is None:

            image_aux_list = [images]

            bs, nimgs_per_sample = input_ids.size(0), image_aux_list[0].size(0) // input_ids.size(0)
            dtype = image_aux_list[0].dtype

            image_aux_list = [_.flatten(0, 1) for _ in image_aux_list]

            si_token_len = self.get_model().config.si_token_len
            miv_token_len = self.get_model().config.miv_token_len

            si_final_height = si_final_width = int(si_token_len**0.5)
            miv_final_height = miv_final_width = int(miv_token_len**0.5)

            if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":

                sharded_images = [xs.enable_manual_sharding(images, ("fsdp", None, None, None)).global_tensor]

                sharded_image_features = self.encode_images(sharded_images)[0]
                
                image_features = xs.disable_manual_sharding(sharded_image_features, ("fsdp", None, None), (images.size(0), sharded_image_features.size(1), sharded_image_features.size(2)), mesh=xs.get_global_mesh()).global_tensor
                # image_aux_features_list = [image_features]
                if hasattr(self, 'generation_alignment_tower_list') and isinstance(self.generation_alignment_tower_list, list) and len(self.generation_alignment_tower_list) > 0:
                    # Mirror SigLIP path: shard images_gen across the SPMD mesh before VAE encode
                    tower = self.generation_alignment_tower_list[0]
                    tower.vae.encoder.gradient_checkpointing = False
                    with torch.no_grad():
                        sharded_images_gen = xs.enable_manual_sharding(images_gen, ("fsdp", None, None, None)).global_tensor
                        
                        sharded_pred_image_features = tower(sharded_images_gen)
                        # sharded_pred_image_features = self.encode_images(sharded_images_gen)


                        pred_image_features = xs.disable_manual_sharding(
                            sharded_pred_image_features,
                            ("fsdp", None, None),
                            (
                                images_gen.size(0),
                                sharded_pred_image_features.size(1),
                                sharded_pred_image_features.size(2),
                            ),
                            mesh=xs.get_global_mesh(),
                        ).global_tensor
                        pred_image_features = pred_image_features.clone().detach()
                        xm.mark_step()

                else:
                    pred_image_features = image_features.clone().detach()



            elif os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_MP":
                image_aux_features_list = self.encode_images(image_aux_list)
            else:
                image_aux_list = [_.to(torch.bfloat16) for _ in image_aux_list]
                
                image_aux_features_list = self.encode_images(image_aux_list)
                image_features = image_aux_features_list[0]

                image_features = image_features.to(device=input_ids.device, dtype=dtype)
                pred_image_features = image_features.clone().detach()
                # raise NotImplementedError
        else:
            # If image_embeds is provided, we assume it is already encoded.
            image_features = image_embeds
            # print("image features shape is", image_features.shape)
            dtype = image_features.dtype

        # --------------------------------------------------------------
        # AR-DDT: add noise to *answer* image patches BEFORE projection
        # --------------------------------------------------------------
        vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')
        arddt_cache = None  # default

        if vision_loss_mode_cfg == "ar-ddt":
            diff_head = getattr(self, 'diff_head', None)
            if diff_head is None:
                raise RuntimeError("AR-DDT requested but diff_head is not initialised.")

            # Generate noisy version of image features for AR-DDT
            # pred_image_features: (B*T_img, feat_dim) - clean features
            x_clean = pred_image_features
            
            x_t, t_ar, x_end = diff_head.add_noise(x_clean)  # Same shape as x_clean
            
            
            # Blend noisy and clean features using answer_img_mask
            # answer_img_mask: (B_small, max_images) - 1 for answer images, 0 for context images
            # x_clean, x_t: (B_img, T_img, feat_dim) where B_img = B_small * max_images
            B_img, T_img, feat_dim = x_clean.shape
            B_small, max_images = answer_img_mask.shape
            tokens_per_image = T_img  # by construction

            # Sanity check dimensions match expectation
            assert B_small * max_images == B_img, (
                f"Mismatch: answer_img_mask expects {B_small*max_images} images but got {B_img} feature batches"
            )

            # Create patch-level mask (B_small, max_images, tokens_per_image)
            valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B_small, max_images, tokens_per_image)  # (B_small, M, P)
            # Flatten to (B_img, T_img)
            patch_mask = valid_mask_patch.reshape(B_img, T_img)  # (B_img, T_img)

            # For debugging

            # Blend: use noisy features for answer images, clean for context images
            patch_mask_expanded = patch_mask.unsqueeze(-1).repeat(1, 1, feat_dim)  # (B_img, T_img, feat_dim)
            blended_features = x_clean + patch_mask_expanded * (x_t - x_clean)
            
            # Cache AR-DDT info for loss computation
            arddt_cache = {
                'x_t': x_t,
                't_ar': t_ar, 
                'x_end': x_end,
                'answer_img_mask': answer_img_mask
            }
            
            # Use blended features for projection
            image_features = blended_features
        
        # @MetaMorph changes: It is important to freeze the features
        # (projection happens once for every mode)
        image_features = self.get_model().mm_projector(image_features).to(dtype)

        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(image_features, xs.get_global_mesh(), ("fsdp", None, None))
        #     xs.mark_sharding(pred_image_features, xs.get_global_mesh(), ("fsdp", None, None))


        

    
        if self.get_model().config.mm_projector_type == 'sva':
            raise NotImplementedError
        else:
            if self.get_model().config.image_aspect_ratio == "anyres":
                # get miv features
                feature_side_len = int(image_features.size(1) ** .5)
                miv_features = image_features.clone().unflatten(1, (feature_side_len, feature_side_len)).permute(0, 3, 1, 2)
                miv_features = F.interpolate(miv_features.float(), size=(miv_final_height, miv_final_width), mode='bilinear', align_corners=False).type_as(miv_features)
                si_features = image_features.clone().permute(0, 2, 1).unflatten(-1, (si_final_height, si_final_width))


        if IS_XLA_AVAILABLE:
            if self.get_model().config.image_aspect_ratio == "anyres":

                miv_features = miv_features.permute(0, 2, 3, 1)
                si_features = si_features.permute(0, 2, 3, 1)
                miv_features = torch.cat([miv_features, self.model.image_newline[None, None, None, :].expand(*miv_features.size()[:2], 1, -1)], dim=2)
                si_features = torch.cat([si_features, self.model.image_newline[None, None, None, :].expand(*si_features.size()[:2], 1, -1)], dim=2)
                miv_features = miv_features.flatten(1, 2).unflatten(0, (bs, nimgs_per_sample)).flatten(1, 2)
                si_features = si_features.flatten(1, 2).unflatten(0, (bs, nimgs_per_sample)).flatten(1, 2)

                image_features = torch.cat([miv_features, si_features], dim=1)

        else:
            if self.get_model().config.image_aspect_ratio == "anyres":

                image_features = image_features.view(bs, final_height, final_width, -1)
                image_features_unpadded = []
                final_size = []
                if self.get_model().config.mm_projector_type == 'sva':
                    vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final = self.rearrange_vision_tower_features_inference(vision_tower_aux_feature_list, final_height, image_sizes, unpad=True)
                    global_context_feature_final = []

                for batch_i in range(bs):
                    cur_image_feature = image_features[batch_i]
                    image_size = image_sizes[batch_i]

                    cur_image_feature = unpad_image(cur_image_feature.unsqueeze(0), image_size)

                    cur_h, cur_w = cur_image_feature.shape[1:3]
                    final_size.append((cur_h, cur_w))
                    cur_image_feature = cur_image_feature.view(1, cur_h, cur_w, -1)
                    cur_image_feature = torch.cat((
                            cur_image_feature,
                            self.model.image_newline.view(1, 1, 1, -1).expand(1, cur_h, 1, -1).to(cur_image_feature.device)
                        ), dim=2)
                    cur_image_feature = cur_image_feature.flatten(1, 2)
                    image_features_unpadded.append(cur_image_feature.squeeze(0))

                    if self.get_model().config.mm_projector_type == 'sva':
                        cur_global_context_feature = global_context_feature[batch_i].expand(cur_h*cur_w, 1, -1)
                        global_context_feature_final.append(cur_global_context_feature)
                if self.get_model().config.mm_projector_type == 'sva':
                    global_context_feature_final = torch.cat(global_context_feature_final, 0)

                image_features = image_features_unpadded

        if self.get_model().config.image_aspect_ratio == "anyres":
            vision_token_indices = vision_token_indices[:, :labels.size(1)].clone()
            image_features = torch.gather(image_features, 1, vision_token_indices.unsqueeze(-1).expand(-1, -1, image_features.size(-1)))


        if IS_XLA_AVAILABLE:

            # embed the input_ids
            new_input_ids_padded_for_emb = torch.where(input_ids==IMAGE_TOKEN_INDEX, 0, input_ids)
            input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb)


            if not self.get_model().embed_tokens.weight.requires_grad: # ! NOTE@shusheng: if embed_tokens is frozen then input_embeds will become a leaf node on which the inplace operation is not allowed
                input_embeds = input_embeds.clone() # ! NOTE@shusheng: clone the tensor to make sure the inplace operation is allowed


                
            # if IS_XLA_AVAILABLE:
            # Create base indices that just select positions sequentially
            batch_size, seq_len = input_ids.shape
            total_images, num_tokens_per_image, feature_dim = image_features.shape
            images_per_batch = total_images // batch_size
            image_feature_dim = pred_image_features.shape[-1]

            image_features = image_features.view(batch_size, images_per_batch*num_tokens_per_image, feature_dim) # (B, L, D)
            pred_image_features = pred_image_features.view(batch_size, images_per_batch*num_tokens_per_image, image_feature_dim) # (B, L, D)

            # Debug image feature shapes
            # dbg_shape("image_features(after encode & proj)", image_features)
            # dbg_shape("pred_image_features(clone)", pred_image_features)

            # Use custom kernel to embed image features into input embeddings
            input_embeds = apply_custom_kernel(input_embeds, image_features, vision_token_indices)

            # dbg_shape("input_embeds(after embed)", input_embeds)

            zero_selected_features = torch.zeros((batch_size, seq_len, image_feature_dim), 
                                    dtype=pred_image_features.dtype, 
                                    device=input_ids.device)

            zero_selected_features = zero_selected_features.clone()
            
            # Similarly for selected features (always use pred_image_features for targets)
            selected_features = apply_custom_kernel(
                zero_selected_features, 
                pred_image_features, 
                vision_token_indices
            )

            # dbg_shape("selected_features", selected_features)
            
            # Get vision loss mode early to handle query mode before truncation
            vision_loss_mode = getattr(self.get_model().config, 'vision_loss_mode', 'causal')
            
            # QUERY MODE: Replace answer image tokens with latent queries BEFORE truncation
            if vision_loss_mode in ("query", "half-query", "query-block"):
                latent_queries = self.get_model().latent_queries
                if latent_queries is not None:
                    expanded_latent_queries = latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
                    expanded_latent_queries = expanded_latent_queries.repeat(1, images_per_batch, 1)

                    # Create answer region mask on FULL sequence
                    full_input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
                    full_is_start_token = (input_ids == self.im_start_id)
                    full_is_end_token = (input_ids == self.im_end_id)
                    full_is_answer_token = torch.cat([
                        (labels != IGNORE_INDEX),
                        torch.zeros(batch_size, input_ids.shape[1] - labels.shape[1], 
                                   dtype=torch.bool, device=labels.device)
                    ], dim=1)

                    # Create answer region mask
                    full_start_in_answer = torch.logical_and(full_is_start_token, full_is_answer_token)
                    full_end_in_answer = torch.logical_and(full_is_end_token, full_is_answer_token)
                    
                    full_region_markers = torch.zeros_like(full_input_embed_mask)
                    full_region_markers = torch.where(full_start_in_answer, torch.ones_like(full_region_markers), full_region_markers)
                    full_region_markers = torch.where(full_end_in_answer, -torch.ones_like(full_region_markers), full_region_markers)
                    full_answer_regions = torch.cumsum(full_region_markers, dim=1) > 0

                    answer_image_mask = torch.logical_and(full_input_embed_mask.bool(), full_answer_regions).float()

                    # dbg_shape("answer_image_mask", answer_image_mask)

                    zero_latent = torch.zeros_like(input_embeds)
                    latent_embedded = apply_custom_kernel(zero_latent, expanded_latent_queries, vision_token_indices)
                    input_embeds = input_embeds + answer_image_mask.unsqueeze(-1) * (latent_embedded - input_embeds)



            elif vision_loss_mode == "ar-ddt":
                # AR-DDT: noisy patches were already blended before projection
                # Just need to prepare the cached info for loss computation
                
                if arddt_cache is None:
                    raise RuntimeError("AR-DDT mode but no cached info - this shouldn't happen")
                
                # The input_embeds already contain the projected blended features
                # We just need to return the cached AR-DDT info for loss computation
                extra_mm_outputs = (
                    arddt_cache['x_t'],           # noisy features (B*T_img, feat_dim)
                    arddt_cache['t_ar'],          # timesteps (B*T_img,)
                    arddt_cache['answer_img_mask'], # answer image mask (B, max_images)
                    pred_image_features,          # clean features (B*T_img, feat_dim)
                    reverse_vti                   # reverse mapping for loss computation
                )

                return None, position_ids, attention_mask, past_key_values, input_embeds, labels, None, None, None, extra_mm_outputs

            input_embeds = input_embeds[:, :labels.size(1)].clone() # discard the padding
            attention_mask = attention_mask[:, :labels.size(1)].clone() # discard the padding

            # Create mask for image tokens
            input_embed_mask = (input_ids == IMAGE_TOKEN_INDEX).int()
            # Clip to match labels length
            selected_features = selected_features[:, :labels.size(1)]
            input_embed_mask = input_embed_mask[:, :labels.size(1)]

            # Find start/end tokens and answer regions (needed for all modes)
            is_start_token = (input_ids[:, :labels.size(1)] == self.im_start_id)
            is_end_token = (input_ids[:, :labels.size(1)] == self.im_end_id)
            is_answer_token = (labels != IGNORE_INDEX)

            # Create answer region mask
            start_in_answer = torch.logical_and(is_start_token, is_answer_token)
            end_in_answer = torch.logical_and(is_end_token, is_answer_token)

            # Use cumulative sum to mark regions between start and end tokens
            region_markers = torch.zeros_like(input_embed_mask)
            region_markers = torch.where(start_in_answer, torch.ones_like(region_markers), region_markers)
            region_markers = torch.where(end_in_answer, -torch.ones_like(region_markers), region_markers)
            answer_regions = torch.cumsum(region_markers, dim=1) > 0
            
            # Final mask - only image tokens in answer regions
            input_embed_mask = torch.logical_and(input_embed_mask.bool(), answer_regions).int()

            # ------------------------------------------------------------------
            # Build attention bias for BLOCK vision-loss mode (bidirectional
            # attention among image tokens). This bias is later consumed by the
            # SPMD flash-attention kernel via `ab=` argument (see
            # `Qwen2Attention.forward`).
            # ------------------------------------------------------------------
            vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')
            attention_bias = None
            if vision_loss_mode_cfg == "block" or vision_loss_mode_cfg == "query-block":
                num_heads = getattr(self.get_model().config, 'num_attention_heads', 1)
                attention_bias = build_block_attention_bias(
                    input_ids=input_ids[:, :labels.size(1)],
                    input_embed_mask=input_embed_mask,  # [B, L]
                    im_start_id=self.im_start_id,
                    num_heads=num_heads,
                    vision_loss_mode=vision_loss_mode_cfg,
                )  # [B, H, L, L]

                # Mark sharding so each head dimension follows the FSDP mesh
                if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
                    xs.mark_sharding(attention_bias, xs.get_global_mesh(), ("fsdp", None, None, None))

            # ------------------------------------------------------------------
            # End of BLOCK attention-bias construction
            # ------------------------------------------------------------------

            extra_mm_outputs = None
            # ---------- Collator provided helpers ----------
            if reverse_vti is not None and answer_img_mask is not None:
                # Sanity-check: gather must reproduce image_features exactly

                extra_mm_outputs = (image_features, reverse_vti, answer_img_mask, pred_image_features)
                return None, position_ids, attention_mask, past_key_values, input_embeds, labels, selected_features, input_embed_mask, attention_bias, extra_mm_outputs
            print("Its in qwen forward!!!!!!!")
            return None, position_ids, attention_mask, past_key_values, input_embeds, labels, selected_features, input_embed_mask, attention_bias, extra_mm_outputs



        else:
            # Let's just add dummy tensors if they do not exist,
            # it is a headache to deal with None all the time.
            # But it is not ideal, and if you have a better idea,
            # please open an issue / submit a PR, thanks.
            _labels = labels
            _position_ids = position_ids
            _attention_mask = attention_mask
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask.bool()
            if position_ids is None:
                position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # remove the padding using attention_mask -- FIXME
            _input_ids = input_ids
            input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

            new_input_embeds = []
            new_labels = []
            cur_image_idx = 0
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    cur_image_idx += 1
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                cur_new_input_embeds = []
                cur_new_labels = []

                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]



                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)

            # Truncate sequences to max length as image embeddings can make the sequence longer
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            # Combine them
            max_len = max(x.shape[0] for x in new_input_embeds)
            batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            if _labels is None:
                new_labels = None
            else:
                new_labels = new_labels_padded

            if _attention_mask is None:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

            if _position_ids is None:
                position_ids = None

            #       position_ids.dtype if position_ids is not None else None, attention_mask.dtype if attention_mask is not None else None, past_key_values.dtype if past_key_values is not None else None,
            #       new_input_embeds.dtype, new_labels.dtype if new_labels is not None else None)
            new_input_embeds = new_input_embeds.to(dtype=torch.bfloat16)

            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, None, None, None

            # Reference from TPU code
            # return None, position_ids, attention_mask, past_key_values, input_embeds, labels, selected_features, input_embed_mask, attention_bias, extra_mm_outputs





    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


            if model_args.tune_adapter_and_vision_head:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                
                # NEW: restrict gradient updates to *only* the rows that correspond to the
                # newly-added special tokens (<im_start>, <im_end>, etc.).  We assume these
                # tokens were appended by `tokenizer.add_tokens` just above, so they occupy
                # the **last `num_new_tokens` rows** of the embedding matrix.
                if num_new_tokens > 0:
                    embedding_weight = self.get_input_embeddings().weight  # nn.Parameter

                    # Make sure this tensor actually requires grad (it might
                    # still be False if a blanket freeze happened earlier).
                    if not embedding_weight.requires_grad:
                        embedding_weight.requires_grad_(True)

                    def _mask_grad(grad, n_new=num_new_tokens):  # noqa: ANN001
                        """Zero gradients for all old tokens, keep new-token grads."""
                        if grad.dim() == 2 and grad.size(0) > n_new:
                            grad = grad.clone()
                            grad[:-n_new] = 0
                        return grad

                    # Register the hook exactly once
                    if not hasattr(embedding_weight, "_grad_mask_hook"):
                        embedding_weight._grad_mask_hook = embedding_weight.register_hook(_mask_grad)

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

            # if model_args.pretrain_adapter_and_vision_head:
            #     mm_projector_weights = torch.load(model_args.pretrain_adapter_and_vision_head, map_location='cpu')
            #     embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            #     assert num_new_tokens == 2
            #     if input_embeddings.shape == embed_tokens_weight.shape:
            #         input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            #     elif embed_tokens_weight.shape[0] == num_new_tokens:
            #         input_embeddings[-num_new_tokens:] = embed_tokens_weight
            #     else:
            #         raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


def dbg_shape(name: str, tensor):  # noqa: ANN001
    """Utility to print a tensor's shape/dtype/device for TPU compile debugging."""
    try:
        print(f"[QDBG] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    except AttributeError:
        # Handle None or non-tensor passed accidentally
        print(f"[QDBG] {name}: {tensor}")