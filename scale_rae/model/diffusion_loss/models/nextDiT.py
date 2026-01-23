# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0
from diffusers.models.embeddings import  get_2d_rotary_pos_embed_lumina
from .next_utils import NextDiTPatchEmbed, NextDiTTimestepCaptionEmbedding, NextDiTTextProjection, NextFFN
from .next_utils import LayerNormContinuous, RMSNormZero, RMSNorm
from scale_rae.utils import IS_XLA_AVAILABLE
import math


# Monkey patch: replace scaled_dot_product_attention with standard attention for TPU
def _patched_LuminaAttnProcessor2_0_call(
    self,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    query_rotary_emb: Optional[torch.Tensor] = None,
    key_rotary_emb: Optional[torch.Tensor] = None,
    base_sequence_length: Optional[int] = None,
) -> torch.Tensor:
    from diffusers.models.embeddings import apply_rotary_emb

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = hidden_states.shape

    # Get Query-Key-Value Pair
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query_dim = query.shape[-1]
    inner_dim = key.shape[-1]
    head_dim = query_dim // attn.heads
    dtype = query.dtype

    # Get key-value heads
    kv_heads = inner_dim // head_dim

    # Apply Query-Key Norm if needed
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, kv_heads, head_dim)
    value = value.view(batch_size, -1, kv_heads, head_dim)

    # Apply RoPE using real ops (XLA-friendly), avoid complex/view ops
    def _rope_cos_sin_expand(freqs: torch.Tensor | tuple, D: int, dtype: torch.dtype, device: torch.device):
        if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
            cos, sin = freqs
        else:
            base = freqs
            if base.dim() == 3 and base.size(0) == 1:
                base = base.squeeze(0)
            cos = torch.cos(base)
            sin = torch.sin(base)
        # Expand last dim to match D if needed (angles given for D//2 pairs)
        if cos.shape[-1] * 2 == D:
            cos = torch.repeat_interleave(cos, 2, dim=-1)
            sin = torch.repeat_interleave(sin, 2, dim=-1)
        # Broadcast to (1, S, 1, D) to match x: (B, S, H, D)
        cos = cos.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
        sin = sin.unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
        return cos, sin

    def _apply_rope_real(x: torch.Tensor, freqs: torch.Tensor | tuple):
        B, S, H, D = x.shape
        cos, sin = _rope_cos_sin_expand(freqs, D, x.dtype, x.device)
        x_f = x.float()
        x_real, x_imag = x_f.reshape(B, S, H, -1, 2).unbind(-1)
        x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x_f * cos) + (x_rot * sin)
        return out.to(x.dtype)

    # Apply TPU-friendly real-valued RoPE
    if query_rotary_emb is not None:
        query = _apply_rope_real(query, query_rotary_emb)
    if key_rotary_emb is not None:
        key = _apply_rope_real(key, key_rotary_emb)

    query, key = query.to(dtype), key.to(dtype)

    # Apply proportional attention if true
    if key_rotary_emb is None:
        softmax_scale = None
    else:
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

    # perform Grouped-qurey Attention (GQA)
    n_rep = attn.heads // kv_heads
    if n_rep >= 1:
        key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

    # Build/expand mask
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, 1, 1, key.shape[1]), device=hidden_states.device, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
    attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)

    # Shapes for attention
    query = query.transpose(1, 2)  # (B, H, Lq, D)
    key = key.transpose(1, 2)      # (B, H, Lk, D)
    value = value.transpose(1, 2)  # (B, H, Lk, D)

    # Standard attention with TPU-safe numerics (compute in float32)
    q32 = query.float()
    k32 = key.float()
    v32 = value.float()
    scores = torch.matmul(q32, k32.transpose(-2, -1))  # (B, H, Lq, Lk) in fp32
    scale_val = (1.0 / math.sqrt(head_dim)) if softmax_scale is None else float(softmax_scale)
    scores = scores * scale_val
    # Use large negative constant in fp32 to avoid -inf on TPU
    neg_large = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
    scores = scores.masked_fill(~attention_mask, neg_large)
    # Stabilize softmax
    scores = scores - scores.max(dim=-1, keepdim=True).values
    attn_probs = torch.softmax(scores, dim=-1)
    # Guard against rows fully masked -> NaNs
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
    out32 = torch.matmul(attn_probs, v32)  # (B, H, Lq, D) in fp32

    hidden_states = out32.transpose(1, 2).to(dtype)
    return hidden_states


# Apply the monkey patch
LuminaAttnProcessor2_0.__call__ = _patched_LuminaAttnProcessor2_0_call

class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock for LuminaNextDiT2DModel.

    Parameters:
        dim (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_kv_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        multiple_of (`int`): The number of multiple of ffn layer.
        ffn_dim_multiplier (`float`): The multipier factor of ffn layer dimension.
        norm_eps (`float`): The eps for norm layer.
        qk_norm (`bool`): normalization for query and key.
        cross_attention_dim (`int`): Cross attention embedding dimension of the input text prompt hidden_states.
        norm_elementwise_affine (`bool`, *optional*, defaults to True),
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        z_channels = 4096, # for crossattn
        norm_eps: float = 1e-5,
        qk_norm: bool  = True,
        norm_elementwise_affine: bool = True,
        class_dropout_prob: float = 0.0, # not actually used
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_heads

        self.gate = nn.Parameter(torch.zeros([num_heads]))

        # Self-attention
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )
        self.attn1.to_out = nn.Identity()

        # Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=z_channels,
            dim_head=dim // num_heads,
            qk_norm="layer_norm_across_heads" if qk_norm else None,
            heads=num_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=LuminaAttnProcessor2_0(),
        )

        self.feed_forward = NextFFN(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.norm1 = RMSNormZero(
            embedding_dim=dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm1_context = RMSNorm(z_channels, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        temb: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
            encoder_hidden_states: (`torch.Tensor`): The hidden_states of text prompt are processed by Gemma encoder.
            encoder_mask (`torch.Tensor`): The hidden_states of text prompt attention mask.
            temb (`torch.Tensor`): Timestep embedding with text prompt embedding.
            cross_attention_kwargs (`Dict[str, Any]`): kwargs for cross attention.
        """
        residual = hidden_states
        # Self-attention
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )
        # Cross-attention
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        cross_attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=None,
            **cross_attention_kwargs,
        )
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        mixed_attn_output = self_attn_output + cross_attn_output
        mixed_attn_output = mixed_attn_output.flatten(-2)
        # linear proj
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)

        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        
        return hidden_states


class LuminaNextDiT2DModel(nn.Module):
    """
    LuminaNextDiT: Diffusion model with a Transformer backbone.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        learn_sigma (`bool`, *optional*, defaults to True):
            Whether the model should learn the sigma parameter, which might be related to uncertainty or variance in
            predictions.
        qk_norm (`bool`, *optional*, defaults to True):
            Indicates if the queries and keys in the attention mechanism should be normalized.
        cross_attention_dim (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 1,
        in_channels: int = 4,
        hidden_size: int = 2304,
        depth: int = 32,  # 32
        num_heads: int = 32,  # 32
        num_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        z_channels: int = 2048,
        scaling_factor: float = 1.0,
        class_dropout_prob: float = 0.0,  # not actually used
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling_factor = scaling_factor
        self.gradient_checkpointing = False

        self.caption_projection = NextDiTTextProjection(in_features=z_channels, hidden_size=hidden_size)
        self.patch_embedder = NextDiTPatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True)

        self.time_caption_embed = NextDiTTimestepCaptionEmbedding(hidden_size=min(hidden_size, 1024), cross_attention_dim=hidden_size) # naming is crazy....

        self.layers = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    hidden_size,
                    norm_eps,
                    qk_norm,
                    norm_elementwise_affine=True,    
                    class_dropout_prob=class_dropout_prob,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = LayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        # self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)
        self.default_2d_rope_embed = get_2d_rotary_pos_embed_lumina(
            self.head_dim,
            384,
            384
        )
        assert (hidden_size // num_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"

        # Initialize weights with TPU-friendly or GPU-friendly routines
        if IS_XLA_AVAILABLE:
            self.initialize_weights()
        else:
            self.GPU_initialize_weights()


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            t (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            y (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """
        if image_rotary_emb is None:
            image_rotary_emb = self.default_2d_rope_embed
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(x, image_rotary_emb)
        image_rotary_emb = image_rotary_emb.to(hidden_states.device)
        # breakpoint()
        #print('y:', y.shape)
        encoder_hidden_states = self.caption_projection(y)
        if encoder_mask is None:
            encoder_mask = torch.ones((y.shape[0], encoder_hidden_states.shape[1]), device=x.device)
        #print('t', t.shape, encoder_hidden_states.shape, encoder_mask.shape, y.shape)
        temb = self.time_caption_embed(t, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()  
        #print('encoder_hidden_states:', encoder_hidden_states.shape, encoder_mask.shape)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                mask,
                image_rotary_emb,
                encoder_hidden_states,
                encoder_mask,
                temb=temb,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        hidden_states = self.norm_out(hidden_states, temb)

        # unpatchify
        height_tokens = width_tokens = self.patch_size
        height, width = img_size[0]
        batch_size = hidden_states.size(0)
        sequence_length = (height // height_tokens) * (width // width_tokens)
        hidden_states = hidden_states[:, :sequence_length].view(
            batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
        )
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        return output

    def initialize_weights(self):
        """Robust TPU-friendly initialization with explicit manual operations."""
        print("--- Starting Manual TPU-friendly Initialization for LuminaNextDiT2DModel ---")

        @torch.no_grad()
        def manual_xavier_uniform(tensor: torch.Tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            random_tensor = torch.empty_like(tensor, dtype=torch.float32).uniform_(0, 1)
            random_tensor = random_tensor * (2 * bound) - bound
            tensor.data.copy_(random_tensor.to(tensor.dtype))
            return tensor

        @torch.no_grad()
        def manual_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02):
            normal_tensor = torch.zeros_like(tensor, dtype=torch.float32)
            normal_tensor.normal_(mean=mean, std=std)
            tensor.data.copy_(normal_tensor.to(tensor.dtype))
            return tensor

        # 1) Initialize all linear layers with manual Xavier, except those we want zero-initialized
        print("Manually initializing linear layers (TPU-safe)...")
        for name, module in self.named_modules():
            try:
                if isinstance(module, nn.Linear):
                    # Skip zero-init targets; will handle below
                    if any(x in name for x in [
                        "norm_out.linear_2",
                        # RMSNormZero.linear lives inside blocks; will handle by type below
                    ]):
                        continue
                    manual_xavier_uniform(module.weight)
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during init of {name}: {e}")

        # 2) Initialize patch embedding explicitly (treat like linear)
        try:
            print("Initializing patch embedding...")
            w = self.patch_embedder.proj.weight.data
            manual_xavier_uniform(w)
            if self.patch_embedder.proj.bias is not None:
                self.patch_embedder.proj.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during patch embedding init: {e}")

        # 3) Initialize caption projection MLP gently
        try:
            print("Initializing caption projection (normal 0.02)...")
            if isinstance(self.caption_projection, NextDiTTextProjection):
                manual_normal_(self.caption_projection.linear_1.weight, std=0.02)
                if self.caption_projection.linear_1.bias is not None:
                    self.caption_projection.linear_1.bias.data.fill_(0.0)
                manual_normal_(self.caption_projection.linear_2.weight, std=0.02)
                if self.caption_projection.linear_2.bias is not None:
                    self.caption_projection.linear_2.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during caption projection init: {e}")

        # 4) Zero-init gating paths to start residuals neutrally
        try:
            print("Zero-initializing RMSNormZero.linear in blocks (gating neutral start)...")
            for i, block in enumerate(self.layers):
                if hasattr(block, "norm1") and isinstance(block.norm1, RMSNormZero):
                    lin = block.norm1.linear
                    if isinstance(lin, nn.Linear):
                        lin.weight.data.fill_(0.0)
                        if lin.bias is not None:
                            lin.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during zero-init of RMSNormZero: {e}")

        # 5) Zero-out final projection so initial output is well-behaved
        try:
            print("Zero-initializing norm_out.linear_2 (final projection)...")
            if isinstance(self.norm_out, LayerNormContinuous) and getattr(self.norm_out, "linear_2", None) is not None:
                self.norm_out.linear_2.weight.data.fill_(0.0)
                if self.norm_out.linear_2.bias is not None:
                    self.norm_out.linear_2.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during zero-init of final projection: {e}")

        # 6) TPU sync if available
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            xm.mark_step()
            print("XLA mark_step after initialization")
        except Exception:
            pass

        # Quick stats
        print("--- Quick Weight Statistics Check ---")
        def check_weight_statistics(name: str, tensor: Optional[torch.Tensor]):
            if tensor is None:
                return
            try:
                t_abs_max = tensor.abs().max().item()
                t_mean = tensor.abs().mean().item()
                t_std = tensor.std().item()
                print(f"{name} - max: {t_abs_max:.6f}, mean: {t_mean:.6f}, std: {t_std:.6f}")
                if tensor.isnan().any() or tensor.isinf().any():
                    print(f"!!! ERROR: {name} has NaN or Inf values !!!")
            except Exception as e:
                print(f"Error checking {name}: {e}")

        # sample checks
        if len(self.layers) > 0:
            first_block = self.layers[0]
            if hasattr(first_block, "norm1") and isinstance(first_block.norm1, RMSNormZero):
                check_weight_statistics("First Block RMSNormZero.linear.weight", first_block.norm1.linear.weight)
        if isinstance(self.norm_out, LayerNormContinuous) and getattr(self.norm_out, "linear_2", None) is not None:
            check_weight_statistics("norm_out.linear_2.weight", self.norm_out.linear_2.weight)
        print("--- TPU-friendly Initialization Complete ---")

    def GPU_initialize_weights(self):
        """Standard GPU-friendly initialization mirroring TPU intent."""
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Explicit patch embed init
        w = self.patch_embedder.proj.weight.data
        nn.init.xavier_uniform_(w)
        if self.patch_embedder.proj.bias is not None:
            nn.init.constant_(self.patch_embedder.proj.bias, 0)

        # Gentle caption projection init
        if isinstance(self.caption_projection, NextDiTTextProjection):
            nn.init.normal_(self.caption_projection.linear_1.weight, std=0.02)
            if self.caption_projection.linear_1.bias is not None:
                nn.init.constant_(self.caption_projection.linear_1.bias, 0)
            nn.init.normal_(self.caption_projection.linear_2.weight, std=0.02)
            if self.caption_projection.linear_2.bias is not None:
                nn.init.constant_(self.caption_projection.linear_2.bias, 0)

        # Zero-init RMSNormZero linear gates
        for block in self.layers:
            if hasattr(block, "norm1") and isinstance(block.norm1, RMSNormZero):
                lin = block.norm1.linear
                if isinstance(lin, nn.Linear):
                    nn.init.constant_(lin.weight, 0)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)

        # Zero-out final projection
        if isinstance(self.norm_out, LayerNormContinuous) and getattr(self.norm_out, "linear_2", None) is not None:
            nn.init.constant_(self.norm_out.linear_2.weight, 0)
            if self.norm_out.linear_2.bias is not None:
                nn.init.constant_(self.norm_out.linear_2.bias, 0)