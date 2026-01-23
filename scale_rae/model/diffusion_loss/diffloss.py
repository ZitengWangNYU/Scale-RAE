from .diffusion.rf import RectifiedFlow
from .models import DiT_ARCH
from .models.AdaLNmlp import SimpleMLPAdaLN
from .diffusion import create_diffusion
from scale_rae.utils import IS_XLA_AVAILABLE

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm  # <-- Import XLA model

import torch.nn as nn
import math
import torch
from typing import Optional

from dataclasses import dataclass, asdict
@dataclass
class DiTkwargs:
    input_size: int = 16
    patch_size: int = 1
    hidden_size: int = 1152
    in_channels: int = 1152
    depth: int = 28
    num_heads: int = 16
    class_dropout_prob: float = .0
    z_channels: int = 4096
    # for next DiT
    # use default by now

def L_to_P(zs:torch.Tensor, split:float = 1)-> torch.Tensor:
    """
    zs: [batch_size, seq_len, hidden_size]
    return: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    """
    # reshape it to square
    batch_size, num_patches, hidden_size = zs.shape
    pn = int(num_patches ** 0.5)
    zs = zs.view(batch_size, pn, pn, hidden_size)
    # zs = self.forward_norm(zs)
    # channel goes first
    # [batch_size, hidden_size, patch_size, patch_size]
    zs = zs.permute(0, 3, 1, 2).contiguous()
    sqrt_split = int(split ** 0.5)
    split_c = int(hidden_size // split)
    split_pn = pn * sqrt_split
    # reshape to bsz, split_c, split_pn, split_pn
    # first split to split_c, sqrt_split, sqrt_split, pn, pn
    zs = zs.view(batch_size, split_c, sqrt_split, sqrt_split, pn, pn)
    # then permute to split_c, split_pn, sqrt_split, split_pn, sqrt_split
    zs = zs.permute(0, 1, 4, 2, 5, 3).contiguous()
    # then reshape to bsz, hidden_size, split_pn, split_pn
    zs = zs.reshape(batch_size, split_c, split_pn, split_pn)
    return zs.contiguous()


def P_to_L(zs: torch.Tensor, split: float = 1) -> torch.Tensor:
    """
    zs: [batch_size, hidden_size//split, sqrt(seq_len*split), sqrt(seq_len*split)]
    return: [batch_size, seq_len, hidden_size]
    """
    batch_size, c, pn, pn = zs.shape
    aggregated_c = c * split
    sqrt_split = int(split ** 0.5)
    split_pn = int(pn // sqrt_split)
    # zs = zs.view(batch_size, c, sqrt_split, split_pn, sqrt_split, split_pn)
    zs = zs.reshape(batch_size, c, split_pn, sqrt_split, split_pn, sqrt_split)
    # try reshape back to see diff
    # do a reverse permute to (0,1,4,2,5,3)
    zs = zs.permute(0, 1, 3, 5, 2, 4).contiguous()
    zs = zs.view(batch_size, aggregated_c, split_pn, split_pn)
    zs = zs.permute(0, 2, 3, 1).contiguous()
    zs = zs.view(batch_size, split_pn, split_pn, aggregated_c)
    zs = zs.view(batch_size, split_pn*split_pn, aggregated_c)
    return zs.contiguous()
from dataclasses import dataclass, asdict
@dataclass
class DiTkwargs:
    input_size: int = 16
    patch_size: int = 1
    hidden_size: int = 1152
    in_channels: int = 1152
    depth: int = 28
    num_heads: int = 16
    class_dropout_prob: float = .0
    z_channels: int = 4096
    # for next DiT
    # use default by now


class RectifiedFlowProjector(nn.Module):
    """
    deprecated
    """
    def __init__(self, z_channels: int, diffusion_channels: int, diffusion_tokens: int, split_per_token: int = 1, inference_step: int = 25, guidance_scale: float = 1.0, cfg_interval: list[float, float] = [-1e4, 1e4], model_hidden_size: int = 1152, model_depth: int = 28, model_heads: int = 16, class_dropout_prob: float = .0, use_mlp: bool = False, base_dim: int = 4096):
        """
        a rectified flow + lightningDiT model, conditioned on z_channels

        input: z: (batch_size, z_channels), x: (batch_size, diffusion_tokens, diffusion_channels)

        will be splitted into x_ : (batch_size, diffusion_tokens * split_per_token, diffusion_channels//split_per_token) for diffusion process
        """
        super().__init__()
        self.z_channels = z_channels
        self.diffusion_channels = diffusion_channels
        self.diffusion_tokens = diffusion_tokens
        self.split_per_token = split_per_token

        self.base_dim = base_dim  # default follows SD3
        input_dim = diffusion_channels * diffusion_tokens
        input_base_dimension_ratio = math.sqrt(
            self.base_dim / input_dim)  # do rescaling
        # hardcode rf config for now
        self.train_flow: RectifiedFlow = create_diffusion(
            str(1000),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            input_base_dimension_ratio=input_base_dimension_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
            use_schedule_shift=True,
            diffusion_kwargs=None,
        )

        self.inference_flow: RectifiedFlow = create_diffusion(
            str(inference_step),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            input_base_dimension_ratio=input_base_dimension_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
            use_schedule_shift=True,
            diffusion_kwargs=None,
        )

        total_tokens = diffusion_tokens * split_per_token
        sqrt_tokens = int(math.sqrt(total_tokens))
        assert sqrt_tokens * \
            sqrt_tokens == total_tokens, f"total tokens {total_tokens} should be a square number"
        input_dim = int(diffusion_channels // split_per_token)
        assert input_dim % 2 == 0, f"input dim {input_dim} should be even for DiT+ RoPE"
        assert split_per_token * \
            input_dim == diffusion_channels, f"split_per_token {split_per_token} * input_dim {input_dim} should be equal to diffusion_channels {diffusion_channels}"
        self.input_size = sqrt_tokens
        if use_mlp:
            raise NotImplementedError("MLP is deprecated")
            assert total_tokens == 1, f"MLP only supports single token"
            self.model = SimpleMLPAdaLN(
                input_size=sqrt_tokens,
                patch_size=1,
                in_channels=input_dim,
                hidden_size=model_hidden_size,
                depth=model_depth,
                class_dropout_prob=class_dropout_prob,
                z_channels=z_channels,
                use_gembed=True,
            )

        else:
            dit_cls = LightningDiT if isinstance(
                model_hidden_size, int) else LightningDDT
            self.model = dit_cls(
                input_size=sqrt_tokens,
                patch_size=1,
                in_channels=input_dim,
                hidden_size=model_hidden_size,
                depth=model_depth,
                num_heads=model_heads,
                z_channels=z_channels,
                class_dropout_prob=class_dropout_prob
            )

        self.use_cfg = guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.cfg_interval = cfg_interval

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        return x  # do nothing, just impl. it for completeness

    def zero_check(self):

        # Debug
        # ---------------------------------------------------------------------
        ZERO_PARAM_PATTERNS = (
            ".adaLN_modulation.1.weight",
            ".adaLN_modulation.1.bias",
            "final_layer.linear.weight",
            "final_layer.linear.bias",
            # "y_proj.weight",
            # "y_proj.bias",
        )

        def _is_zero_target(name: str) -> bool:
            return any(key in name for key in ZERO_PARAM_PATTERNS)

        # ---------------------------------------------------------------------
        # 2)  Helper to zero them *after* the model is on the XLA device
        def zero_condition_path(model):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if _is_zero_target(n):
                        p.zero_()
            xm.mark_step()     # flush to the TPU

        # ---------------------------------------------------------------------
        # 3)  Assert they really are zero (ignores all other params)
        def assert_zeros(model, verbose: bool = True):
            offending = []
            for name, p in model.named_parameters():
                if p is None or not _is_zero_target(name):
                    continue                     # skip params that are allowed non-zero
                if (p.abs() > 0).any().item():  # forces evaluation on TPU
                    offending.append(name)
                    if verbose and xm.is_master_ordinal():
                        print(
                            f"❌ {name} not zero, max={p.abs().max().item():.4e}")
            return offending

        # ---------------------------------------------------------------------
        # 4)  Example call-site – put this right after replication
        # NOTE:  assume `self.model` is your SimpleMLPAdaLN instance

        off = assert_zeros(self.model)                  # verify
        if xm.is_master_ordinal():
            print("!!!! Second zero-check →",
                  "OK !!!!" if not off else f"{len(off)} params non-zero")

        xm.mark_step()

    def add_noise(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Public helper that adds RF noise to a clean patch tensor **without**
        touching gradients.  This lets higher-level code (e.g. AR-DDT) build
        the exact same noisy sample that the decoder will later denoise.

        Args
        -----
        x : Tensor (B, L, C)
            Clean SigLIP patches in **L-layout** (token list).
        t : Optional Tensor (B,) in [0,1]
            Timestep to use.  If None we sample it the same way the legacy
            training_loss path does.

        Returns
        -------
        x_t : Tensor (B, L, C)
            Noisy version of *x* at time *t* (still in L-layout).
        t    : Tensor (B,)
            The timestep actually used (identical to the input if given).
        x_end: Tensor (B, L, C)
            The random terminal state used to build the velocity target –
            needed when the caller wants *v_target = x_end − x*.
        """
        # Convert list-layout → patch-grid for the internal diffusion utils
        x_p = L_to_P(x, self.split_per_token)  # (B, C, H, W)

        if t is None:
            t = self.train_flow.get_timestep(x_p)  # (B,)

        # Retrieve α(t), σ(t) and build a matching x_end (pure noise)
        alpha_t = self.train_flow.get_alphas(t)
        sigma_t = self.train_flow.get_sigmas(t)
        x_end = self.train_flow.get_x_end(x_p.shape, device=x.device)

        # Add noise using the same q_sample implementation as in training_losses
        x_t_p = self.train_flow.q_sample(x_p, alpha_t, sigma_t, x_end)
        x_t = P_to_L(x_t_p, self.split_per_token)  # back to list layout

        x_end_L = P_to_L(x_end, self.split_per_token)
        return x_t, t, x_end_L

    def training_loss(self, z: torch.Tensor, x: torch.Tensor, t: Optional[torch.Tensor] = None, x_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute RF loss.

        Legacy usage (unchanged):
            training_loss(z, x)           # t and x_t generated internally

        New AR-DDT usage:
            x_t, t, _ = add_noise(x)
            training_loss(z, x, t=t, x_t=x_t)
        """
        assert x.shape[1] == self.diffusion_tokens, f"x shape {x.shape} should be equal to diffusion_tokens {self.diffusion_tokens}"
        assert x.shape[2] == self.diffusion_channels, f"x shape {x.shape} should be equal to diffusion_channels {self.diffusion_channels}"

        # Convert to patch grid
        x_p = L_to_P(x, self.split_per_token)

        if t is None:
            # —— original path ——
            loss = self.train_flow.training_losses(
                self.model, x_p, None, model_kwargs={"y": z})
            return loss["loss"]

        # —— AR-DDT path ——
        assert x_t is not None, "When providing t you must also provide x_t"
        x_t_p = L_to_P(x_t, self.split_per_token)
        loss = self.train_flow.training_losses(
            self.model,
            x_p,
            t,
            model_kwargs={"y": z},
            x_t=x_t_p,
        )
        return loss["loss"]

    @torch.no_grad()
    def infer(self, z: torch.Tensor, x_end: torch.Tensor = None, guidance_level=None) -> torch.Tensor:
        if guidance_level is not None:
            #     f"[DEBUG] Using guidance_level {guidance_level} instead of self.guidance_scale {self.guidance_scale}")
            self.guidance_scale = guidance_level
            self.use_cfg = guidance_level > 1.0

        guidance_scale = self.guidance_scale

        cfg_interval = self.cfg_interval
        n = z.shape[0]
        x_end_shape = (n, self.model.in_channels,
                       self.input_size, self.input_size)
        if x_end is None:
            x_end = self.inference_flow.get_x_end(
                x_end_shape, device=z.device, dtype=z.dtype)
        if self.use_cfg:
            x_end = torch.cat([x_end, x_end], dim=0)
            model_kwargs = dict(y=z, cfg_scale=guidance_scale,
                                cfg_interval=cfg_interval)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=z)
            sample_fn = self.model.forward
        samples = self.inference_flow.p_sample_loop(
            sample_fn,
            x_end_shape,
            x_end,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=z.device,
        )

        if self.use_cfg:
            # remove unconditioned samples
            samples, _ = samples.chunk(2, dim=0)

        samples = P_to_L(samples, self.split_per_token)

        return samples


# def create_rf_projector(model_kwargs:dict) -> RectifiedFlowProjector:
#     """
#     Create a RectifiedFlowProjector instance with dummy parameters.
#     """
#     # Dummy parameters for testing
#     return RectifiedFlowProjector(
#         **model_kwargs,
#         inference_step=2
#     )


class FullSequenceRectifiedFlowProjector(nn.Module):
    def __init__(self, z_channels: int, diffusion_channels: int, diffusion_tokens: int, inference_step: int = 25, guidance_scale: float = 1.0, cfg_interval: list[float, float] = [-1e4, 1e4], model_hidden_size: int = 1152, model_depth: int = 28, model_heads: int = 16, class_dropout_prob: float = .0, use_mlp: bool = False, batchnorm_path: str = None, base_dim: int = 4096, **model_kwargs):
        """
        a rectified flow + lightningDiT model, conditioned on z_channels

        input: z: (batch_size, z_channels), x: (batch_size, diffusion_tokens, diffusion_channels)

        will be splitted into x_ : (batch_size, diffusion_tokens * split_per_token, diffusion_channels//split_per_token) for diffusion process
        """
        super().__init__()
        self.z_channels = z_channels
        self.diffusion_channels = diffusion_channels
        self.diffusion_tokens = diffusion_tokens
        self.base_dim = base_dim  # following SD3
        input_dim = diffusion_channels * diffusion_tokens
        input_base_dimension_ratio = math.sqrt(
            self.base_dim / input_dim)  # do rescaling
        # hardcode rf config for now
        self.train_flow: RectifiedFlow = create_diffusion(
            str(1000),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            input_base_dimension_ratio=input_base_dimension_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
            use_schedule_shift=True,
            diffusion_kwargs=None,
        )

        self.inference_flow: RectifiedFlow = create_diffusion(
            str(inference_step),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            input_base_dimension_ratio=input_base_dimension_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
            use_schedule_shift=True,
            diffusion_kwargs=None,
        )
        dit_cls=model_kwargs.pop('dit_cls', 'DiT')
        if not isinstance(model_hidden_size, int):
            dit_cls = 'DDT'
        DiT_kwargs = DiTkwargs(
            input_size=int(math.sqrt(diffusion_tokens)),
                in_channels=diffusion_channels,
                patch_size = 1,
            hidden_size=model_hidden_size,
            depth=model_depth,
            num_heads=model_heads,
            class_dropout_prob=class_dropout_prob,
            z_channels=z_channels,
        )
        total_tokens = diffusion_tokens
        sqrt_tokens = int(math.sqrt(total_tokens))
        assert sqrt_tokens * \
            sqrt_tokens == total_tokens, f"total tokens {total_tokens} should be a square number"
        input_dim = int(diffusion_channels)
        self.input_size = sqrt_tokens
        if use_mlp:
            assert total_tokens == 1, f"MLP only supports single token"
            self.model = SimpleMLPAdaLN(
                input_size=sqrt_tokens,
                patch_size=1,
                in_channels=input_dim,
                hidden_size=model_hidden_size,
                depth=model_depth,
                class_dropout_prob=class_dropout_prob,
                z_channels=z_channels,
                use_gembed=True,
            )

        else:
            if dit_cls == 'xattnDiT':
                # LuminaNextDiT2DModel has a strict signature; do not forward extraneous kwargs
                # Only pass the core DiT parameters it expects
                print("We popped out the extra kwargs for xattnDiT:", model_kwargs.keys())
                self.model = DiT_ARCH[dit_cls](
                    **asdict(DiT_kwargs)
                )
            else:
                self.model = DiT_ARCH[dit_cls](
                    **asdict(DiT_kwargs),
                    **model_kwargs,  # allow extra kwargs for LightningDiT
                )
        self.use_cfg = guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.cfg_interval = cfg_interval

        if batchnorm_path is not None:
            # load batchnorm stats
            self.load_batchnorm_stats(batchnorm_path)
            print(
                f"[DEBUG] data_std: {self.data_std.shape}, data_mean: {self.data_mean.shape}")
            # exit()

        self.normalize_data = batchnorm_path is not None


    def load_batchnorm_stats(self, batchnorm_path: str):
        """
        Load batchnorm stats from a file.
        """
        bn_state_dict = torch.load(batchnorm_path, map_location="cpu")
        bn_var, bn_mean = bn_state_dict["running_var"], bn_state_dict["running_mean"]
        bn_std = torch.sqrt(bn_var + 1e-5)

        # register as buffer
        self.register_buffer("data_std", bn_std)
        self.register_buffer("data_mean", bn_mean)

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        return x  # do nothing, just impl. it for completeness

    def zero_check(self):

        # Debug
        # ---------------------------------------------------------------------
        ZERO_PARAM_PATTERNS = (
            ".adaLN_modulation.1.weight",
            ".adaLN_modulation.1.bias",
            "final_layer.linear.weight",
            "final_layer.linear.bias",
            # "y_proj.weight",
            # "y_proj.bias",
        )

        def _is_zero_target(name: str) -> bool:
            return any(key in name for key in ZERO_PARAM_PATTERNS)

        # ---------------------------------------------------------------------
        # 2)  Helper to zero them *after* the model is on the XLA device
        def zero_condition_path(model):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if _is_zero_target(n):
                        p.zero_()
            xm.mark_step()     # flush to the TPU

        # ---------------------------------------------------------------------
        # 3)  Assert they really are zero (ignores all other params)
        def assert_zeros(model, verbose: bool = True):
            offending = []
            for name, p in model.named_parameters():
                if p is None or not _is_zero_target(name):
                    continue                     # skip params that are allowed non-zero
                if (p.abs() > 0).any().item():  # forces evaluation on TPU
                    offending.append(name)
                    if verbose and xm.is_master_ordinal():
                        print(
                            f"❌ {name} not zero, max={p.abs().max().item():.4e}")
            return offending

        # ---------------------------------------------------------------------
        # 4)  Example call-site – put this right after replication
        # NOTE:  assume `self.model` is your SimpleMLPAdaLN instance

        off = assert_zeros(self.model)                  # verify
        if xm.is_master_ordinal():
            print("!!!! Second zero-check →",
                  "OK !!!!" if not off else f"{len(off)} params non-zero")

        xm.mark_step()

    def add_noise(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        Public helper that adds RF noise to a clean patch tensor **without**
        touching gradients.  This lets higher-level code (e.g. AR-DDT) build
        the exact same noisy sample that the decoder will later denoise.

        Args
        -----
        x : Tensor (B, L, C)
            Clean SigLIP patches in **L-layout** (token list).
        t : Optional Tensor (B,) in [0,1]
            Timestep to use.  If None we sample it the same way the legacy
            training_loss path does.

        Returns
        -------
        x_t : Tensor (B, L, C)
            Noisy version of *x* at time *t* (still in L-layout).
        t    : Tensor (B,)
            The timestep actually used (identical to the input if given).
        x_end: Tensor (B, L, C)
            The random terminal state used to build the velocity target –
            needed when the caller wants *v_target = x_end − x*.
        """
        # For FullSequenceRectifiedFlowProjector, we work directly in L-layout
        # (no split_per_token conversion needed)

        if t is None:
            # Convert to patch grid format for get_timestep
            x_p = x.permute(0, 2, 1).contiguous()  # (B, C, L)
            # (B, C, H, W)
            x_p = x_p.view(x_p.shape[0], x_p.shape[1],
                           self.input_size, self.input_size)
            t = self.train_flow.get_timestep(x_p)  # (B,)

        # Get RF parameters
        alpha_t = self.train_flow.get_alphas(t)
        sigma_t = self.train_flow.get_sigmas(t)

        # Generate x_end (pure noise) in the same shape as x
        x_end = torch.randn_like(x)  # (B, L, C)

        # Add noise: x_t = alpha_t * x + sigma_t * x_end
        alpha_t = alpha_t.view(-1, 1, 1)  # (B, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1)  # (B, 1, 1)
        x_t = alpha_t * x + sigma_t * x_end

        return x_t, t, x_end

    def training_loss(self, z: torch.Tensor, x: torch.Tensor, t: torch.Tensor = None, x_t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute training loss for FullSequenceRectifiedFlowProjector.

        Args:
            z: Condition tensor (B, z_channels)
            x: Clean input tensor (B, L, C) in L-layout
            t: Optional timestep tensor (B,) - if None, will be sampled
            x_t: Optional noisy input tensor (B, L, C) - if None, will be generated from x

        Returns:
            loss: Scalar loss tensor
        """
        assert x.shape[1] == self.diffusion_tokens, f"x shape {x.shape} should be equal to diffusion_tokens {self.diffusion_tokens}"
        assert x.shape[2] == self.diffusion_channels, f"x shape {x.shape} should be equal to diffusion_channels {self.diffusion_channels}"

        # x: [B, L, D]
        # first convert to [B, H, W, C]
        H = W = int(math.sqrt(self.diffusion_tokens))
        assert H * \
            W == self.diffusion_tokens, f"diffusion_tokens {self.diffusion_tokens} should be a square number"

        if self.normalize_data:
            # normalize the data
            data_mean = self.data_mean.to(x.device).unsqueeze(
                0).expand(x.shape[0], *x.shape[1:])
            data_std = self.data_std.to(x.device).unsqueeze(
                0).expand(x.shape[0], *x.shape[1:])
            x = (x - data_mean) / data_std

        # Convert to patch grid format for diffusion training
        x = x.view(x.shape[0], H, W, self.diffusion_channels).permute(
            0, 3, 1, 2).contiguous()

        # Handle external t and x_t (for AR-DDT mode)
        if x_t is not None:
            # AR-DDT mode: x_t provided externally, convert to patch grid format
            if self.normalize_data:
                # normalize x_t the same way as x
                data_mean = self.data_mean.to(x_t.device).unsqueeze(
                    0).expand(x_t.shape[0], *x_t.shape[1:])
                data_std = self.data_std.to(x_t.device).unsqueeze(
                    0).expand(x_t.shape[0], *x_t.shape[1:])
                x_t = (x_t - data_mean) / data_std
            x_t = x_t.view(x_t.shape[0], H, W, self.diffusion_channels).permute(
                0, 3, 1, 2).contiguous()

        # Get timestep
        t = self.train_flow.get_timestep(x) if t is None else t

        # Compute loss
        if x_t is not None:
            # AR-DDT mode: use external x_t
            loss = self.train_flow.training_losses(
                self.model, x, t, model_kwargs={"y": z}, x_t=x_t)
        else:
            # Regular mode: let training_losses generate x_t internally
            loss = self.train_flow.training_losses(
                self.model, x, t, model_kwargs={"y": z})

        return loss["loss"]

    # (legacy single-argument version removed – unified implementation above)

    @torch.no_grad()
    def infer(self, z: torch.Tensor, x_end: torch.Tensor = None, guidance_level=None) -> torch.Tensor:
        if guidance_level is not None:
            #     f"[DEBUG] Using guidance_level {guidance_level} instead of self.guidance_scale {self.guidance_scale}")
            self.guidance_scale = guidance_level
            self.use_cfg = guidance_level > 1.0
        guidance_scale = self.guidance_scale
        cfg_interval = self.cfg_interval
        n = z.shape[0]
        x_end_shape = (n, self.model.in_channels,
                       self.input_size, self.input_size)

        if x_end is None:
            x_end = self.inference_flow.get_x_end(
                x_end_shape, device=z.device, dtype=z.dtype)
        if self.use_cfg:
            # raise NotImplementedError("CFG is not supported (yet)")
            x_end = torch.cat([x_end, x_end], dim=0)
            model_kwargs = dict(y=z, cfg_scale=guidance_scale,
                                cfg_interval=cfg_interval)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=z)
            sample_fn = self.model.forward
        samples = self.inference_flow.p_sample_loop(
            sample_fn,
            x_end_shape,
            x_end,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=z.device,
        )

        if self.use_cfg:
            # remove unconditioned samples
            samples, _ = samples.chunk(2, dim=0)
        # convert back to [B, L, D]
        samples = samples.permute(0, 2, 3, 1).contiguous()
        samples = samples.view(
            samples.shape[0], -1, self.diffusion_channels)  # [B, L, D]
        if self.normalize_data:
            # denormalize the data
            data_mean = self.data_mean.to(samples.device).unsqueeze(
                0).expand(samples.shape[0], *samples.shape[1:])
            data_std = self.data_std.to(samples.device).unsqueeze(
                0).expand(samples.shape[0], *samples.shape[1:])
            samples = samples * data_std + data_mean
        return samples


def create_rf_projector(model_kwargs: dict) -> RectifiedFlowProjector:
    """
    Create a RectifiedFlowProjector instance with dummy parameters.
    """
    # Dummy parameters for testing
    return FullSequenceRectifiedFlowProjector(  # updated to use FullSequenceRectifiedFlowProjector rather than RectifiedFlowProjector
        **model_kwargs,
        inference_step=50
    )
