
import torch.nn as nn
from .model_utils import GaussianFourierEmbedding, RMSNorm
import torch
from torch.nn import LayerNorm
from typing import *
class NextDiTTimestepCaptionEmbedding(nn.Module):
    def __init__(self, hidden_size=4096, cross_attention_dim=2048, frequency_embedding_size=256):
        super().__init__()
        # self.time_proj = Timesteps(
        #     num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        # )
        # use Gaussian Embedding for continous flow matching
        self.timestep_embedder = GaussianFourierEmbedding(hidden_size=hidden_size, embedding_size=frequency_embedding_size)

        self.caption_embedder = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(
                cross_attention_dim,
                hidden_size,
                bias=True,
            ),
        )

    def forward(self, timestep, caption_feat, caption_mask):
        # timestep embedding:
        time_embed = self.timestep_embedder(timestep)
        # caption condition embedding:
        caption_mask_float = caption_mask.float().unsqueeze(-1)
        caption_feats_pool = (caption_feat * caption_mask_float).sum(dim=1) / caption_mask_float.sum(dim=1)
        caption_feats_pool = caption_feats_pool.to(caption_feat)
        caption_embed = self.caption_embedder(caption_feats_pool)
        conditioning = time_embed + caption_embed

        return conditioning

class NextDiTPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for Lumina-T2X

    Args:
        patch_size (`int`, defaults to `2`): The size of the patches.
        in_channels (`int`, defaults to `4`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
    """

    def __init__(self, patch_size=2, in_channels=4, embed_dim=768, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=embed_dim,
            bias=bias,
        )

    def forward(self, x, freqs_cis):
        """
        Patchifies and embeds the input tensor(s).

        Args:
            x (List[torch.Tensor] | torch.Tensor): The input tensor(s) to be patchified and embedded.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]: A tuple containing the patchified
            and embedded tensor(s), the mask indicating the valid patches, the original image size(s), and the
            frequency tensor(s).
        """
        freqs_cis = freqs_cis.to(x[0].device)
        patch_height = patch_width = self.patch_size
        batch_size, channel, height, width = x.size()
        height_tokens, width_tokens = height // patch_height, width // patch_width

        x = x.view(batch_size, channel, height_tokens, patch_height, width_tokens, patch_width).permute(
            0, 2, 4, 1, 3, 5
        )
        x = x.flatten(3)
        x = self.proj(x)
        x = x.flatten(1, 2)

        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

        return (
            x,
            mask,
            [(height, width)] * batch_size,
            freqs_cis[:height_tokens, :width_tokens].flatten(0, 1).unsqueeze(0),
        )

class NextDiTTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class RMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_eps: float, norm_elementwise_affine: bool):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])

        return x, gate_msa, scale_mlp, gate_mlp

class LayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # The norm layer can be configured to have scale and shift parameters.
        # AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class NextFFN(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
        multiple_of (`int`, *optional*): Value to ensure hidden dimension is a multiple
            of this value.
        ffn_dim_multiplier (float, *optional*): Custom multiplier for hidden
            dimension. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        # self.silu = FP32SiLU()
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.linear_2(self.silu(self.linear_1(x)) * self.linear_3(x))