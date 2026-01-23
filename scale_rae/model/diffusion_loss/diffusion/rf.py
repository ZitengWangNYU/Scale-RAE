from math import sqrt
from tqdm import tqdm
from .gaussian_diffusion import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from typing import Optional

def logclip(t, eps: float = 1e-20):
    return torch.log(t.clamp(min = eps))
class RectifiedFlow(GaussianDiffusion):
    def __init__(
        self,
        *,
        size_ratio: float = 1.,
        schedule: ScheduleType.LOGIT_NORMAL,
        pred_term = ModelMeanType.VELOCITY,
        loss_type = LossType.WEIGHTED_MSE,
        diffusion_steps: int = 1000,
        used_timesteps: list[int] = None,
        logit_normal_alpha : float = 0.,
        logit_normal_beta: float = 1., # default to rf/lognorm(0, 1)
        noise_scale_path : str = None, # optionally scale the noise
        override_ratio: float = None, # optionally override the size_ratio
    ):
        assert isinstance(schedule, ScheduleType), f'Invalid schedule type {schedule}'
        assert isinstance(pred_term, ModelMeanType), f'Invalid pred_term type {pred_term}'
        assert isinstance(loss_type, LossType), f'Invalid loss_type {loss_type}'
        self.schedule = schedule
        self.pred_term = pred_term
        self.loss_type = loss_type
        self.size_ratio = size_ratio
        if override_ratio is not None:
            override_ratio = 1 / sqrt(override_ratio) # override_ratio is the squared size_ratio
            self.size_ratio = override_ratio
        self.log_ratio = math.log(self.size_ratio)
        self.diffusion_steps = diffusion_steps
        self.lognorm_mean = logit_normal_alpha 
        self.lognorm_std = logit_normal_beta
        used_timesteps = set(timestep  for timestep in used_timesteps) if used_timesteps is not None else set(range(0, diffusion_steps)) # default to all timesteps
        # sort the timesteps in ascending order
        self.used_timesteps = sorted(used_timesteps)
        self.sampler_timesteps = self.get_sampler_timesteps(diffusion_steps)
        if noise_scale_path is not None:
            noise_scale = torch.load(noise_scale_path) # should be a bn
            self.noise_std = (noise_scale.running_var + noise_scale.eps).sqrt() # this should be a tensor of shape [C]
            self.noise_mean = noise_scale.running_mean # this should be a tensor of shape [C]
        else:
            self.noise_std = self.noise_mean = None
        #print(f'used_timesteps: {self.used_timesteps}, sampler_timesteps: {self.sampler_timesteps}')
        #print(f'lognorm_mean: {self.lognorm_mean}, lognorm_std: {self.lognorm_std}')
        #print(f'mapped used_timesteps: {[self.sampler_timesteps[i] for i in self.used_timesteps]}')
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        unnormalize x to [0, 1]
        """
        if self.noise_mean is not None:
            C = x.shape[1] # assume x is [batch_size, C, H, W]
            noise_mean = self.noise_mean.view(1, C, *( 1 for _ in range(len(x.shape) - 2))) # broadcast to [1, C, 1, ...]
            noise_mean = noise_mean.to(x.device)
            noise_std = self.noise_std.view(1, C, *( 1 for _ in range(len(x.shape) - 2)))
            noise_std = noise_std.to(x.device)
            x = x * noise_std + noise_mean
        return x
    def get_x_end(self, x_shape, device, dtype=torch.bfloat16) -> torch.Tensor:
        """
        get x_end (noise)
        """
        x_end = torch.randn(x_shape, device= device, dtype= dtype) # Line 70

        if self.noise_std is not None:
            x_end = self.unnormalize(x_end) # use pre-defined norm and mean as initial noise
        return x_end
    def q_sample(self, x_start, alpha_t, sigma_t, x_end= None) -> torch.Tensor:
        if x_end is None:
            x_end = self.get_x_end(x_start.shape, x_start.device)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        x_t = alpha_t * x_start + sigma_t * x_end

        return x_t 
    def get_timestep(self, x_start: torch.Tensor) -> int:
        batch_shape = x_start.size(0)
        if self.schedule == ScheduleType.LINEAR:
            t = torch.rand(batch_shape, device=batch_shape.device).to(x_start.dtype)
        elif self.schedule == ScheduleType.LOGIT_NORMAL:
            u = torch.normal(self.lognorm_mean, self.lognorm_std, size=(batch_shape,), device=x_start.device).to(x_start.dtype)
            t = torch.sigmoid(u)
            # BEGIN DEBUG
            # For debugging purposes, set the timestep tensor `t` to all ones while preserving
            # its shape, device, and dtype. Comment out the following line to restore the
            # original behaviour based on the logit-normal schedule.
            # NOTE: This forces the diffusion process to always use t = 1.
            # -----------------------------------------------------------------------------
            # t = torch.ones_like(torch.sigmoid(u))
            
            # t = u
            
            # I suspect it is because sigmoid
            # t = torch.rand(batch_shape, device=batch_shape.device).to(x_start.dtype)
            
        else:
            raise NotImplementedError(f'Invalid schedule type {self.schedule}')
        # do a resolution-based shifting
        sqrt_size_ratio = 1 / self.size_ratio # already sqrted
        t = sqrt_size_ratio * t / (1 + (sqrt_size_ratio - 1) * t)
        return t
    
    def get_sampler_timesteps(self, total_timesteps: int) -> list[int]:
        if self.schedule == ScheduleType.LINEAR:
            t = np.linspace(1/ total_timesteps, 1, total_timesteps) # LOGIT_NORMAL sampling doesn't work bro
        elif self.schedule == ScheduleType.LOGIT_NORMAL:
            #print(f'using logit-normal sampling')
            # we assume the current t is in uniform [0, 1], now we should map it to N(0, 1)
            x = np.linspace(1/ total_timesteps, 1, total_timesteps) # uniform [0, 1]
            #u = np.sqrt(2) * erfinv(2 * x - 1)
            #t = 1 / (1 + np.exp(-(self.lognorm_mean + self.lognorm_std * u))) # map to logit-normal
            t = x
        else:
            raise NotImplementedError(f'Invalid schedule type {self.schedule}')
        sqrt_size_ratio = 1 / self.size_ratio # already sqrted
        t = sqrt_size_ratio * t / (1 + (sqrt_size_ratio - 1) * t) # do a resolution-based shifting
        # convert to list
        t = t.tolist()
        return t
    def get_sigmas(self, t: torch.Tensor) -> torch.Tensor:
        return t
    def get_alphas(self, t: torch.Tensor) -> torch.Tensor:
        # rf formulation: x_t = sigma_t * x_1 (noise) + (1 - sigma_t) * x_0
        return 1 - t

    def training_losses(self, model: nn.Module, x_start: torch.Tensor, t: Optional[torch.Tensor] = None, model_kwargs=None, x_end=None, x_t: Optional[torch.Tensor] = None):
        if x_end is None:
            x_end = self.get_x_end(x_start.shape, x_start.device)
        if t is None:
            t = self.get_timestep(x_start)  # legacy random sampling
        else:
            assert t.dtype is not torch.long and t.dtype is not torch.int, f'Invalid t dtype {t.dtype}'

        if x_t is None:
            # original behaviour: build noisy sample internally
            alpha_t = self.get_alphas(t)
            sigma_t = self.get_sigmas(t)
            z_t = self.q_sample(x_start, alpha_t, sigma_t, x_end)
        else:
            # AR-DDT: noisy sample already supplied in same layout as x_start
            z_t = x_t
            # Still need alpha_t and sigma_t for model input and target
            alpha_t = self.get_alphas(t)
            sigma_t = self.get_sigmas(t)

#
        ###### Start of Debug ######

        model_pred = model(z_t, sigma_t, **model_kwargs)

        
        if self.pred_term == ModelMeanType.VELOCITY: # SID uses mse loss for velocity
            target = x_end - x_start
            eps_pred = model_pred
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')

        ###### End of Debug ######
        
        mse_target = (eps_pred - target) ** 2
        #weight = self.get_weight(t)


        #loss = mean_flat(weight * mse_target)
        
        loss = mean_flat(mse_target) # use mse loss for now, can be changed to weighted mse later
        
        terms = {
            'mse': mean_flat(mse_target),
            'loss': loss,
        }
        return terms

        
    def get_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        get w^(λ_t)
        by default weighting, w^(λ_t) = p(λ_t) , in eps prediction loss
        the returned weight is divided by p(λ_t) so you can directly use get_weight(t) * (pred - eps) ** 2
        """
        #lambda_t = self.logsnr_t(t, self.schedule).view(-1, 1, 1, 1).to(t.device)
        if self.loss_type == LossType.WEIGHTED_MSE:
            # do sigmoid weighting
            weight_t = torch.ones_like(t).view(-1, 1, 1, 1).to(t.device)
            return weight_t
        elif self.loss_type == LossType.MSE: 
            return torch.ones_like(t).view(-1, 1, 1, 1).to(t.device)
        else:
            raise NotImplementedError(f'Invalid loss type {self.loss_type}')
    def euler_sample(self, model, x_t, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs):
        sigma_t = self.get_sigmas(u_t)
        sigma_s = self.get_sigmas(u_s)
        #print(f'sigma_t: {sigma_t}, sigma_s: {sigma_s}, model_kwargs: {model_kwargs}')
        model_pred = model(x_t, sigma_t, **model_kwargs)
        if denoised_fn is not None:
            model_pred = denoised_fn(model_pred)
        delta_t = sigma_s - sigma_t # < 0
        delta_t = delta_t.view(-1, 1, 1, 1).to(x_t.device)
        x_s = x_t + delta_t * model_pred
        #print(f'sample from timestep {u_t} to {u_s}, sigma_t: {sigma_t}, sigma_s: {sigma_s}, delta_t: {delta_t}')
        return x_s
    def x_pred_from_x_t(self, model, x_t, u_t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        """
        x_pred = x_t + (sigma_s - sigma_t) * model_pred
        """
        sigma_t = self.get_sigmas(u_t)
        sigma_s = self.get_sigmas(0)
        # sigma_s = 0 for x_pred
        delta_t = sigma_s - sigma_t # < 0
        delta_t = delta_t.view(-1, 1, 1, 1).to(x_t.device)
        model_pred = model(x_t, sigma_t, **model_kwargs)
        if denoised_fn is not None:
            model_pred = denoised_fn(model_pred)
        x_pred = x_t + delta_t * model_pred
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)
        return x_pred
    def p_sample_loop(self, model, shape, x_end=None, clip_denoised=False, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
        if x_end is None:
            x_end = self.get_x_end(shape, device)
        x_t = x_end.to(device)
        progess_bar = tqdm(reversed(range(1, len(self.used_timesteps)))) if progress else reversed(range(1, len(self.used_timesteps)))
        for i in progess_bar:
            #print(f'Processing timestep {i}:{self.used_timesteps[i]} to timestep {i - 1}:{self.used_timesteps[i - 1]}')
            #u_t = self.used_timesteps[i] / self.diffusion_steps # current t
            #u_s = self.used_timesteps[i - 1] / self.diffusion_steps # next t
            u_t = self.sampler_timesteps[self.used_timesteps[i]]
            u_s = self.sampler_timesteps[self.used_timesteps[i - 1]]
            u_t = torch.tensor(u_t).to(device).repeat(x_t.size(0)).to(x_t.dtype) # repeat for batch size
            u_s = torch.tensor(u_s).to(device).repeat(x_t.size(0)).to(x_t.dtype)
            x_s = self.euler_sample(model, x_t, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs)
            x_t = x_s 
            #print(f'sample from timestep {i} to {i - 1}')
            if IS_XLA_AVAILABLE:
                xm.mark_step()
        if clip_denoised:
            x_pred = x_t.clamp(-1, 1)
        else:
            x_pred = x_t
        return x_pred
        