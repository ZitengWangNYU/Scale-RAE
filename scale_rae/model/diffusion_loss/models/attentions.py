import torch.nn as nn
import torch
from timm.models.vision_transformer import Attention

def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (`int`):
            Size of query q.
        k_size (`int`):
            Size of key k.
        rel_pos (`torch.Tensor`):
            Relative position embeddings (num_embeddings, num_channels).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel position embeddings.
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_relative_positions(attn, queries, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings as introduced in
    [MViT2](https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py).

    Args:
        attn (`torch.Tensor`):
            Attention map.
        queries (`torch.Tensor`):
            Query q in the attention layer with shape (batch_size, queries_height * queries_width, num_channels).
        rel_pos_h (`torch.Tensor`):
            Relative position embeddings (Lh, num_channels) for height axis.
        rel_pos_w (`torch.Tensor`):
            Relative position embeddings (Lw, num_channels) for width axis.
        q_size (`Tuple[int]`):
            Spatial sequence size of query q with (queries_height, queries_width).
        k_size (`Tuple[int]`):
            Spatial sequence size of key k with (keys_height, keys_width).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)

    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
    relative_height = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
    relative_weight = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)

    attn = (
        attn.view(batch_size, queries_height, queries_width, keys_height, keys_width)
        + relative_height[:, :, :, :, None]
        + relative_weight[:, :, :, None, :]
    ).view(batch_size, queries_height * queries_width, keys_height * keys_width)

    return attn
def window_partition(hidden_state, window_size):
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        hidden_state (`torch.Tensor`):
            Input tokens with [batch_size, height, width, num_channels].
        window_size (`int`):
            Window size.

    Returns:
        `tuple(torch.FloatTensor)` comprising various elements:
        - windows: windows after partition with [batch_size * num_windows, window_size, window_size, num_channels].
        - (padded_height, padded_width): padded height and width before partition
    """
    batch_size, height, width, num_channels = hidden_state.shape

    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size

    # Noop in case pad_width == 0 and pad_height == 0.
    hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))

    padded_height, padded_width = height + pad_height, width + pad_width

    hidden_state = hidden_state.view(
        batch_size, padded_height // window_size, window_size, padded_width // window_size, window_size, num_channels
    )
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows, (padded_height, padded_width)


def window_unpartition(windows, window_size, pad_height_width, height_width):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (`torch.Tensor`):
            Input tokens with [batch_size * num_windows, window_size, window_size, num_channels].
        window_size (`int`):
            Window size.
        pad_height_width (`Tuple[int]`):
            Padded height and width (padded_height, padded_width).
        height_width (`Tuple[int]`):
            Original height and width before padding.

    Returns:
        hidden_state: unpartitioned sequences with [batch_size, height, width, num_channels].
    """
    padded_height, padded_width = pad_height_width
    height, width = height_width
    batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    hidden_state = windows.view(
        batch_size, padded_height // window_size, padded_width // window_size, window_size, window_size, -1
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden_state = hidden_state.view(batch_size, padded_height, padded_width, -1)

    # We always have height <= padded_height and width <= padded_width
    hidden_state = hidden_state[:, :height, :width, :].contiguous()
    return hidden_state


class LocalAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self,
        hidden_size: int,
        num_heads: int,
        input_size: int,
        qkv_bias: bool = False,
        use_relative_position_embeddings: bool = False,
        window_size: int = 4,
    ):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            input_size (`Tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__()

        dim = hidden_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_relative_position_embeddings = use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
        self.window_size = window_size
    def core_attn(self, hidden_state, output_attentions=False):
        batch_size, height, width, _ = hidden_state.shape
        # qkv with shape (3, batch_size, num_heads, height * width, num_channels)
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # queries, keys and values have shape (batch_size * num_heads, height * width, num_channels)
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)

        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)

        if self.use_relative_position_embeddings:
            attention_scores = add_decomposed_relative_positions(
                attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attention_probs = attention_scores.softmax(dim=-1)

        hidden_state = attention_probs @ values
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)
        hidden_state = self.proj(hidden_state)

        if output_attentions:
            attention_probs = attention_probs.reshape(
                batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1]
            )
            outputs = (hidden_state, attention_probs)
        else:
            outputs = (hidden_state,)

        return outputs
    def forward(self, x, output_attentions=False):
        """
        x: [N, T, D]
        first convert to [N, H, W, D], do window partitioning, attention, unpartitioning
        then convert back to [N, T, D]
        """
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, -1))
        #print('before partition', x.shape, 'window size', self.window_size)
        xs, pad_hw = window_partition(x, self.window_size)
        #print('before attn', xs.shape, pad_hw)
        attn_outputs = self.core_attn(
            xs,
            output_attentions=output_attentions
        )
        #print('after attn', attn_outputs[0].shape)
        x = window_unpartition(attn_outputs[0], self.window_size, pad_hw, (h, w))
        #print('after unpartition', x.shape)
        # reshape back to [N, T, D]
        x = x.reshape(shape=(x.shape[0], -1, x.shape[-1]))
        #print('after reshape', x.shape)
        return x