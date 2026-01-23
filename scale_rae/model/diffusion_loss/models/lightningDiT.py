import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp
from ..diffusion import create_diffusion
import torch.nn.functional as F
from .model_utils import TimestepEmbedder, GaussianFourierEmbedding, LabelEmbedder, modulate, get_2d_sincos_pos_embed, ConditionEmbedder, gate
from .attentions import * # Attention and localAttention
# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi, sqrt
import math
import os
import torch
from torch import nn
from scale_rae.utils import IS_XLA_AVAILABLE

from einops import rearrange, repeat
from .model_utils import VisionRotaryEmbeddingFast, SwiGLUFFN, RMSNorm

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm # <-- Import XLA model
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.custom_kernel import flash_attention  # Add this line


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q).to(q.dtype) # rope changes dtype to f32, cast it back
            k = rope(k).to(k.dtype) # rope changes dtype to f32, cast it back

        #  # ADD XLA/SPMD OPTIMIZATION HERE
        # if IS_XLA_AVAILABLE and os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     # Use XLA flash attention with SPMD sharding
        #     q = q.to(v.dtype)
        #     k = k.to(v.dtype)

        #     # print("q, k, v shapes are:", q.shape, k.shape, v.shape)
            
        #     x = flash_attention(
        #         q, k, v, 
        #         causal=False,  # Vision transformers typically don't use causal attention
        #         sm_scale=self.scale,
        #         partition_spec=('fsdp', None, None, None)  # SPMD sharding specification
        #     )
        # elif self.fused_attn:
        #     q = q.to(v.dtype)
        #     k = k.to(v.dtype) # rope may change the q,k's dtype
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True, 
        use_rmsnorm=True,
        wo_shift=False,
        z_channel:int = 4096,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(z_channel, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(z_channel, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + gate(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class LightningFinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, z_channels,  use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_channels, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=4,
        hidden_size = 1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.,
        z_channels=4096,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed: bool = True,    
        xt_normalize: bool = False,
        use_DDT: bool = False,  # whether to use DDT
        DDT_encoder_depth: int = -1,  # depth of the DDT encoder
        cond_silu: bool = True,  # whether to inject t into the DDT decoder
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_gembed = use_gembed
        self.xt_normalize = xt_normalize
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(z_channels) if not use_gembed else GaussianFourierEmbedding(z_channels)
        self.y_embedder = ConditionEmbedder(z_channels, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
        self.use_DDT = use_DDT
        self.DDT_encoder_depth = DDT_encoder_depth
        if self.use_DDT: # the z_channel for the decoder blocks should be the same as the hidden size of DiT
            assert DDT_encoder_depth > 0, "DDT_encoder_depth should be greater than 0 when use_DDT is True"
            assert DDT_encoder_depth < depth, "DDT_encoder_depth should be less than depth"
            block_z_channels = [z_channels] * DDT_encoder_depth + [hidden_size] * (depth - DDT_encoder_depth + 1) # 1 for final layer
        else:
            block_z_channels = [z_channels] * (depth + 1) # 1 for final layer
        self.dit_blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     z_channel=block_z_channels[i],
                     ) for i in range(depth)
        ])
        self.final_layer = LightningFinalLayer(hidden_size, patch_size, self.out_channels, block_z_channels[-1], use_rmsnorm=use_rmsnorm)
        if self.use_DDT:
            self.s_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)# additional embedding for DDT for re-injecting xt
        self.cond_silu = cond_silu # whether to inject t into the DDT decoder, default is False
        if IS_XLA_AVAILABLE:
            self.initialize_weights() # only do this when XLA is available, to avoid import issues
        else:
            self.GPU_initialize_weights()
    def initialize_weights(self):
        """Robust TPU-friendly initialization with explicit manual operations."""
        print("--- Starting Manual TPU-friendly Initialization for LightningDiT ---")
        
        # --- Helper function to manually implement xavier_uniform_ ---
        @torch.no_grad()
        def manual_xavier_uniform(tensor):
            """Manually implement xavier_uniform_ in a TPU-friendly way."""
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            # Generate uniform values between -bound and bound
            with torch.no_grad():
                # Create tensor with values in [0, 1]
                random_tensor = torch.empty_like(tensor, dtype=torch.float32).uniform_(0, 1)
                # Scale to [-bound, bound]
                random_tensor = random_tensor * (2 * bound) - bound
                # Use copy_ for reliable TPU behavior
                tensor.data.copy_(random_tensor.to(tensor.dtype))
            
            return tensor
        
        # --- Helper function for normal initialization ---
        @torch.no_grad()
        def manual_normal_(tensor, mean=0.0, std=1.0):
            """Manually implement normal_ in a TPU-friendly way."""
            with torch.no_grad():
                normal_tensor = torch.zeros_like(tensor, dtype=torch.float32)
                normal_tensor.normal_(mean=mean, std=std)
                tensor.data.copy_(normal_tensor.to(tensor.dtype))
            return tensor
        
        # --- Initialize transformer layers with xavier uniform ---
        print("Manually initializing linear layers...")
        for name, module in self.named_modules():
            try:
                if isinstance(module, nn.Linear):
                    # Skip specific layers that need zero init (will handle them later)
                    if any(x in name for x in ["adaLN_modulation[-1]", "final_layer.linear", "final_layer.adaLN_modulation[-1]"]):
                        print(f"Skipping xavier init for zero-init layer: {name}")
                        continue
                    
                    # Apply manual xavier to weight
                    print(f"Applying manual xavier init to: {name}")
                    manual_xavier_uniform(module.weight)
                    
                    # Zero out bias
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during init of {name}: {e}")
        
        # Force synchronization
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        print("XLA mark_step after manual xavier init")
        
        # --- Initialize pos_embed by sin-cos embedding ---
        try:
            print("Initializing positional embedding...")
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            pos_embed_tensor = torch.from_numpy(pos_embed).unsqueeze(0).to(self.pos_embed.dtype)
            self.pos_embed.data.copy_(pos_embed_tensor)
        except Exception as e:
            print(f"Error during positional embedding init: {e}")
        
        # --- Initialize patch_embed like nn.Linear ---
        try:
            print("Initializing patch embedding...")
            w = self.x_embedder.proj.weight.data
            w_reshaped = w.view([w.shape[0], -1])
            manual_xavier_uniform(w_reshaped)
            if self.x_embedder.proj.bias is not None:
                self.x_embedder.proj.bias.data.fill_(0.0)


            if self.use_DDT: # also initialize the additional patch_embed for DDT
                print("Initializing DDT patch embedding...")
                S = self.s_embedder.proj.weight.data
                S_reshaped = S.view([S.shape[0], -1])
                manual_xavier_uniform(S_reshaped)
                if self.s_embedder.proj.bias is not None:
                    self.s_embedder.proj.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during patch embedding init: {e}")

        if IS_XLA_AVAILABLE:
            xm.mark_step()
        print("XLA mark_step after embedding initializations")
        
        # --- Initialize timestep embedding MLP ---
        try:
            print("Initializing timestep embedding MLP with normal(0, 0.02)...")
            if hasattr(self.t_embedder, 'mlp') and isinstance(self.t_embedder.mlp, nn.Sequential):
                if len(self.t_embedder.mlp) > 0 and isinstance(self.t_embedder.mlp[0], nn.Linear):
                    manual_normal_(self.t_embedder.mlp[0].weight, mean=0.0, std=0.02)
                
                if len(self.t_embedder.mlp) > 2 and isinstance(self.t_embedder.mlp[2], nn.Linear):
                    manual_normal_(self.t_embedder.mlp[2].weight, mean=0.0, std=0.02)
        except Exception as e:
            print(f"Error initializing timestep embedding MLP: {e}")
        
        if IS_XLA_AVAILABLE:
            xm.mark_step()
        print("XLA mark_step after time embedding init")
        
        # --- Zero-out adaLN modulation layers in transformer blocks ---
        print("Zero-initializing adaLN modulation layers...")
        with torch.no_grad():
            for i, block in enumerate(self.dit_blocks):
                try:
                    if isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) > 0:
                        last_layer = block.adaLN_modulation[-1]
                        if isinstance(last_layer, nn.Linear):
                            print(f"Zero-initializing dit_blocks[{i}].adaLN_modulation[-1]")
                            last_layer.weight.data.fill_(0.0)
                            if last_layer.bias is not None:
                                last_layer.bias.data.fill_(0.0)
                except Exception as e:
                    print(f"Error during zero-init of dit_blocks[{i}]: {e}")
        
        # --- Zero-out final layer weights and biases ---
        print("Zero-initializing final layer components...")
        with torch.no_grad():
            try:
                if isinstance(self.final_layer.adaLN_modulation, nn.Sequential) and len(self.final_layer.adaLN_modulation) > 0:
                    last_layer = self.final_layer.adaLN_modulation[-1]
                    if isinstance(last_layer, nn.Linear):
                        print("Zero-initializing final_layer.adaLN_modulation[-1]")
                        last_layer.weight.data.fill_(0.0)
                        if last_layer.bias is not None:
                            last_layer.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.adaLN_modulation: {e}")
            
            try:
                print("Zero-initializing final_layer.linear")
                self.final_layer.linear.weight.data.fill_(0.0)
                if self.final_layer.linear.bias is not None:
                    self.final_layer.linear.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.linear: {e}")
        
        # Force final synchronization
        if IS_XLA_AVAILABLE:
            xm.mark_step()        
            print("XLA mark_step after zero-init")
        
        # --- Validation check ---
        print("\n--- Quick Weight Statistics Check ---")
        def check_weight_statistics(name, tensor):
            if tensor is None:
                return
            
            try:
                tensor_abs_max = tensor.abs().max().item()
                tensor_mean = tensor.abs().mean().item()
                tensor_std = tensor.std().item()
                print(f"{name} - max: {tensor_abs_max:.6f}, mean: {tensor_mean:.6f}, std: {tensor_std:.6f}")
                
                if tensor_abs_max > 10.0:
                    print(f"!!! WARNING: {name} has very large values (max={tensor_abs_max:.6f}) !!!")
                
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"!!! ERROR: {name} has NaN or Inf values !!!")
            except Exception as e:
                print(f"Error checking {name}: {e}")
        
        # Check a sample of weights from different parts of the model
        if len(self.dit_blocks) > 0:
            check_weight_statistics("First DIT Block attention.proj.weight", self.dit_blocks[0].attn.proj.weight)
            check_weight_statistics("First DIT Block adaLN_modulation[-1].weight", self.dit_blocks[0].adaLN_modulation[-1].weight)
        
        check_weight_statistics("final_layer.linear.weight", self.final_layer.linear.weight)
        check_weight_statistics("pos_embed", self.pos_embed)
        
        print("--- TPU-friendly Initialization Complete ---")


    def GPU_initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.use_DDT: # also initialize the additional patch_embed for DDT
            w = self.s_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.s_embedder.proj.bias, 0)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.dit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of condition 
        use_checkpoint: boolean to toggle checkpointing
        """
        s = self.x_embedder(x) + self.pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2 TODO: check x.dtype
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)


        
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(t, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(y, xs.get_global_mesh(), ("fsdp", None, None))
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None))

        #print(f"[BEFORE], x shape: {x.shape}, t shape: {t.shape}, y shape: {y.shape}")
        if len(t.shape) < len(y.shape):
            # y: [B, L, D], t: [B, D], expand t to match y's shape
            input_t = t.unsqueeze(1).expand(-1, y.shape[1], -1)
        c = input_t + y                                # (N, D)
        #print(f"c shape: {c.shape}, x shape: {x.shape}, t shape: {t.shape}, y shape: {y.shape}")
        if self.cond_silu:
            c = nn.functional.silu(c)

        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(c, xs.get_global_mesh(), ("fsdp", None, None))

        if self.use_DDT:
            # use DDT forward
            encoder_blocks = self.dit_blocks[:self.DDT_encoder_depth]
            decoder_blocks = self.dit_blocks[self.DDT_encoder_depth:]
            for block in encoder_blocks:
                s = block(s, c, feat_rope=self.feat_rope)
            if self.cond_silu:
                s = nn.functional.silu(s)
            x = self.s_embedder(x) # no need for additional pos embed
            for block in decoder_blocks:
                x = block(x, s, feat_rope=self.feat_rope)          
            x = self.final_layer(x, s)  
        else:
            for block in self.dit_blocks:
                s = block(s, c, feat_rope=self.feat_rope)
            x = s 
            # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None))
            x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #         xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None))

        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #         xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None, None))

        # if self.learn_sigma:
        #     x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(-1e4, -1e4), interval_cfg: float = 0.0): 
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # FIX: Create a 3D unconditional embedding to match y's shape [B, L, D]
        uncond_embedding = self.y_embedder.dropout_embedding
        # Reshape the unconditional embedding from [D] to [1, 1, D] and expand to [B, L, D]
        learned_embed = uncond_embedding.unsqueeze(0).unsqueeze(0).expand(y.shape[0], y.shape[1], -1)
        # --- END: MODIFICATION ---

        y = torch.cat([y, learned_embed], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        #eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        t = t[0] # check if t < cfg_interval
        if t > cfg_interval[0] and t < cfg_interval[1]:
            if interval_cfg > 1.0:
                half_eps = uncond_eps  + interval_cfg * (cond_eps - uncond_eps)
            else:
                half_eps = cond_eps # only use conditional generation
        else:
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class LightningDDT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=4,
        hidden_size = [1152, 1152],
        depth=[22, 6],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.,
        z_channels=4096,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed: bool = True,    
        cond_silu: bool = True,  # whether to inject t into the DDT decoder
    ):
        print("Initializing LightningDDT model...")
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_gembed = use_gembed
        self.enc_hidden_size,  self.dec_hidden_size = hidden_size
        self.encoder_depth, self.decoder_depth = depth
        self.encoder_num_heads, self.decoder_num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, self.enc_hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(z_channels) if not use_gembed else GaussianFourierEmbedding(z_channels)
        self.y_embedder = ConditionEmbedder(z_channels, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.enc_hidden_size), requires_grad=False)
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            #half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.enc_rope = VisionRotaryEmbeddingFast(
                dim = self.enc_hidden_size // self.encoder_num_heads // 2,
                pt_seq_len=hw_seq_len,
            )
            self.dec_rope = VisionRotaryEmbeddingFast(
                dim= self.dec_hidden_size // self.decoder_num_heads // 2,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.enc_rope = None
            self.dec_rope = None
        
        assert self.encoder_depth > 0, "DDT_encoder_depth should be greater than 0 when use_DDT is True"
        block_z_channels = [z_channels] * self.encoder_depth + [self.dec_hidden_size] * (self.decoder_depth) # 1 for final layer
        hidden_sizes = [self.enc_hidden_size] * self.encoder_depth + [self.dec_hidden_size] * self.decoder_depth
        num_heads = [self.encoder_num_heads] * self.encoder_depth + [self.decoder_num_heads] * self.decoder_depth
        self.dit_blocks = nn.ModuleList([
            LightningDiTBlock(hidden_sizes[i], 
                     num_heads[i], 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     z_channel=block_z_channels[i],
                     ) for i in range(self.encoder_depth + self.decoder_depth)
        ])
        self.s_projector = nn.Linear(
            self.enc_hidden_size, self.dec_hidden_size) if self.enc_hidden_size != self.dec_hidden_size else nn.Identity()
        self.final_layer = LightningFinalLayer(self.dec_hidden_size, patch_size, self.out_channels, self.dec_hidden_size, use_rmsnorm=use_rmsnorm)
        self.s_embedder = PatchEmbed(input_size, patch_size, in_channels, self.dec_hidden_size, bias=True)# additional embedding for DDT for re-injecting xt
        self.cond_silu = cond_silu # whether to inject t into the DDT decoder, default is False
        if IS_XLA_AVAILABLE:
            self.initialize_weights() # only do this when XLA is available, to avoid import issues
        else:
            self.GPU_initialize_weights()
    def initialize_weights(self):
        """Robust TPU-friendly initialization with explicit manual operations."""
        print("--- Starting Manual TPU-friendly Initialization for LightningDiT ---")
        
        # --- Helper function to manually implement xavier_uniform_ ---
        @torch.no_grad()
        def manual_xavier_uniform(tensor):
            """Manually implement xavier_uniform_ in a TPU-friendly way."""
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            # Generate uniform values between -bound and bound
            with torch.no_grad():
                # Create tensor with values in [0, 1]
                random_tensor = torch.empty_like(tensor, dtype=torch.float32).uniform_(0, 1)
                # Scale to [-bound, bound]
                random_tensor = random_tensor * (2 * bound) - bound
                # Use copy_ for reliable TPU behavior
                tensor.data.copy_(random_tensor.to(tensor.dtype))
            
            return tensor
        
        # --- Helper function for normal initialization ---
        @torch.no_grad()
        def manual_normal_(tensor, mean=0.0, std=1.0):
            """Manually implement normal_ in a TPU-friendly way."""
            with torch.no_grad():
                normal_tensor = torch.zeros_like(tensor, dtype=torch.float32)
                normal_tensor.normal_(mean=mean, std=std)
                tensor.data.copy_(normal_tensor.to(tensor.dtype))
            return tensor
        
        # --- Initialize transformer layers with xavier uniform ---
        print("Manually initializing linear layers...")
        for name, module in self.named_modules():
            try:
                if isinstance(module, nn.Linear):
                    # Skip specific layers that need zero init (will handle them later)
                    if any(x in name for x in ["adaLN_modulation[-1]", "final_layer.linear", "final_layer.adaLN_modulation[-1]"]):
                        print(f"Skipping xavier init for zero-init layer: {name}")
                        continue
                    
                    # Apply manual xavier to weight
                    print(f"Applying manual xavier init to: {name}")
                    manual_xavier_uniform(module.weight)
                    
                    # Zero out bias
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during init of {name}: {e}")
        
        # Force synchronization
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        print("XLA mark_step after manual xavier init")
        
        # --- Initialize pos_embed by sin-cos embedding ---
        try:
            print("Initializing positional embedding...")
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            pos_embed_tensor = torch.from_numpy(pos_embed).unsqueeze(0).to(self.pos_embed.dtype)
            self.pos_embed.data.copy_(pos_embed_tensor)
        except Exception as e:
            print(f"Error during positional embedding init: {e}")
        
        # --- Initialize patch_embed like nn.Linear ---
        try:
            print("Initializing patch embedding...")
            w = self.x_embedder.proj.weight.data
            w_reshaped = w.view([w.shape[0], -1])
            manual_xavier_uniform(w_reshaped)
            if self.x_embedder.proj.bias is not None:
                self.x_embedder.proj.bias.data.fill_(0.0)


            print("Initializing DDT patch embedding...")
            S = self.s_embedder.proj.weight.data
            S_reshaped = S.view([S.shape[0], -1])
            manual_xavier_uniform(S_reshaped)
            if self.s_embedder.proj.bias is not None:
                self.s_embedder.proj.bias.data.fill_(0.0)
        except Exception as e:
            print(f"Error during patch embedding init: {e}")

        if IS_XLA_AVAILABLE:
            xm.mark_step()
        print("XLA mark_step after embedding initializations")
        
        # --- Initialize timestep embedding MLP ---
        try:
            print("Initializing timestep embedding MLP with normal(0, 0.02)...")
            if hasattr(self.t_embedder, 'mlp') and isinstance(self.t_embedder.mlp, nn.Sequential):
                if len(self.t_embedder.mlp) > 0 and isinstance(self.t_embedder.mlp[0], nn.Linear):
                    manual_normal_(self.t_embedder.mlp[0].weight, mean=0.0, std=0.02)
                
                if len(self.t_embedder.mlp) > 2 and isinstance(self.t_embedder.mlp[2], nn.Linear):
                    manual_normal_(self.t_embedder.mlp[2].weight, mean=0.0, std=0.02)
        except Exception as e:
            print(f"Error initializing timestep embedding MLP: {e}")
        
        if IS_XLA_AVAILABLE:
            xm.mark_step()
        print("XLA mark_step after time embedding init")
        
        # --- Zero-out adaLN modulation layers in transformer blocks ---
        print("Zero-initializing adaLN modulation layers...")
        with torch.no_grad():
            for i, block in enumerate(self.dit_blocks):
                try:
                    if isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) > 0:
                        last_layer = block.adaLN_modulation[-1]
                        if isinstance(last_layer, nn.Linear):
                            print(f"Zero-initializing dit_blocks[{i}].adaLN_modulation[-1]")
                            last_layer.weight.data.fill_(0.0)
                            if last_layer.bias is not None:
                                last_layer.bias.data.fill_(0.0)
                except Exception as e:
                    print(f"Error during zero-init of dit_blocks[{i}]: {e}")
        
        # --- Zero-out final layer weights and biases ---
        print("Zero-initializing final layer components...")
        with torch.no_grad():
            try:
                if isinstance(self.final_layer.adaLN_modulation, nn.Sequential) and len(self.final_layer.adaLN_modulation) > 0:
                    last_layer = self.final_layer.adaLN_modulation[-1]
                    if isinstance(last_layer, nn.Linear):
                        print("Zero-initializing final_layer.adaLN_modulation[-1]")
                        last_layer.weight.data.fill_(0.0)
                        if last_layer.bias is not None:
                            last_layer.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.adaLN_modulation: {e}")
            
            try:
                print("Zero-initializing final_layer.linear")
                self.final_layer.linear.weight.data.fill_(0.0)
                if self.final_layer.linear.bias is not None:
                    self.final_layer.linear.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.linear: {e}")
        
        # Force final synchronization
        if IS_XLA_AVAILABLE:
            xm.mark_step()        
            print("XLA mark_step after zero-init")
        
        # --- Validation check ---
        print("\n--- Quick Weight Statistics Check ---")
        def check_weight_statistics(name, tensor):
            if tensor is None:
                return
            
            try:
                tensor_abs_max = tensor.abs().max().item()
                tensor_mean = tensor.abs().mean().item()
                tensor_std = tensor.std().item()
                print(f"{name} - max: {tensor_abs_max:.6f}, mean: {tensor_mean:.6f}, std: {tensor_std:.6f}")
                
                if tensor_abs_max > 10.0:
                    print(f"!!! WARNING: {name} has very large values (max={tensor_abs_max:.6f}) !!!")
                
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"!!! ERROR: {name} has NaN or Inf values !!!")
            except Exception as e:
                print(f"Error checking {name}: {e}")
        
        # Check a sample of weights from different parts of the model
        if len(self.dit_blocks) > 0:
            check_weight_statistics("First DIT Block attention.proj.weight", self.dit_blocks[0].attn.proj.weight)
            check_weight_statistics("First DIT Block adaLN_modulation[-1].weight", self.dit_blocks[0].adaLN_modulation[-1].weight)
        
        check_weight_statistics("final_layer.linear.weight", self.final_layer.linear.weight)
        check_weight_statistics("pos_embed", self.pos_embed)
        
        print("--- TPU-friendly Initialization Complete ---")


    def GPU_initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.dit_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, D) tensor of condition 
        use_checkpoint: boolean to toggle checkpointing
        """
        s = self.x_embedder(x) + self.pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2 TODO: check x.dtype
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(t, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(y, xs.get_global_mesh(), ("fsdp", None, None))
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None))

        #print(f"[BEFORE], x shape: {x.shape}, t shape: {t.shape}, y shape: {y.shape}")
        if len(t.shape) < len(y.shape):
            # y: [B, L, D], t: [B, D], expand t to match y's shape
            input_t = t.unsqueeze(1).expand(-1, y.shape[1], -1)
        c = input_t + y                                # (N, D)
        #print(f"c shape: {c.shape}, x shape: {x.shape}, t shape: {t.shape}, y shape: {y.shape}")
        if self.cond_silu:
            c = nn.functional.silu(c)

        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(c, xs.get_global_mesh(), ("fsdp", None, None))
        encoder_blocks = self.dit_blocks[:self.encoder_depth]
        decoder_blocks = self.dit_blocks[self.encoder_depth:]
        for block in encoder_blocks:
            s = block(s, c, feat_rope=self.enc_rope)
        if self.cond_silu:
            s = nn.functional.silu(s)
        s = self.s_projector(s)
        x = self.s_embedder(x)
        for block in decoder_blocks:
            x = block(x, s, feat_rope=self.dec_rope)          
        x = self.final_layer(x, s)  
        
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #         xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None, None))

        # if self.learn_sigma:
        #     x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(-1e4, -1e4), interval_cfg: float = 0.0): 
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # FIX: Create a 3D unconditional embedding to match y's shape [B, L, D]
        uncond_embedding = self.y_embedder.dropout_embedding
        # Reshape the unconditional embedding from [D] to [1, 1, D] and expand to [B, L, D]
        learned_embed = uncond_embedding.unsqueeze(0).unsqueeze(0).expand(y.shape[0], y.shape[1], -1)
        # --- END: MODIFICATION ---

        y = torch.cat([y, learned_embed], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        #eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        t = t[0] # check if t < cfg_interval
        if t > cfg_interval[0] and t < cfg_interval[1]:
            if interval_cfg > 1.0:
                half_eps = uncond_eps  + interval_cfg * (cond_eps - uncond_eps)
            else:
                half_eps = cond_eps # only use conditional generation
        else:
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)