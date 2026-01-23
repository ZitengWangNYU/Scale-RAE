from tqdm import tqdm
from .gaussian_diffusion import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

def logclip(t, eps: float = 1e-20):
    return torch.log(t.clamp(min = eps))
class SimpleDiffusion(GaussianDiffusion):
    def __init__(
        self,
        *,
        size_ratio: float = 1.,
        schedule: ScheduleType.SHIFTED_CONSINE,
        pred_term = ModelMeanType.EPSILON,
        loss_type = LossType.WEIGHTED_MSE,
        diffusion_steps: int = 1000,
        used_timesteps: list[int] = None,
    ):
        assert isinstance(schedule, ScheduleType), f'Invalid schedule type {schedule}'
        assert isinstance(pred_term, ModelMeanType), f'Invalid pred_term type {pred_term}'
        assert isinstance(loss_type, LossType), f'Invalid loss_type {loss_type}'
        self.schedule = schedule
        self.pred_term = pred_term
        self.loss_type = loss_type
        self.size_ratio = size_ratio
        self.log_ratio = math.log(size_ratio)
        self.diffusion_steps = diffusion_steps
        used_timesteps = set(timestep + 1 for timestep in used_timesteps) if used_timesteps is not None else set(range(1, diffusion_steps + 1)) # default to all timesteps
        # sort the timesteps in ascending order
        self.used_timesteps = sorted(used_timesteps)
        
    def logsnr_t(self, t: Union[torch.Tensor, float], schedule: ScheduleType, log_min: float = -15, log_max: float=15) -> torch.Tensor:
        """
            return the log SNR at time t
            logSNR_t = - 2 log (tan(pi * t / 2)) # use consine schedule
            if shifted cosine, logSNR_t_shifted = logSNR_t + 2 self.log_ratio
        """
        logsnr_max = log_max + self.log_ratio
        logsnr_min = log_min - self.log_ratio
        t_min = math.atan(math.exp(logsnr_min / 2)) 
        t_max = math.atan(math.exp(logsnr_max / 2))
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t) # avoid copy construct
        t_boundary = t_min + (t_max - t_min) * t
        logsnr_t = -2 * torch.log(torch.tan(t_boundary))
        if schedule == ScheduleType.SHIFTED_CONSINE:
            logsnr_t += 2 * self.log_ratio
        return logsnr_t
    
    def q_sample(self, x_start, alpha_t, sigma_t, noise= None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return alpha_t * x_start + sigma_t * noise
    def get_x_end(self, x_shape, device):
        return torch.randn(x_shape, device=device)

    def training_losses(self, model: nn.Module, x_start: torch.Tensor, t:torch.Tensor = None, model_kwargs=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        if t is None:
            t = torch.rand(x_start.size(0), device=x_start.device) # sample t from uniform distribution
        else:
            assert t.dtype is not torch.long and t.dtype is not torch.int, f'Invalid t dtype {t.dtype}'
        logsnr_t = self.logsnr_t(t, self.schedule)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x_start.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x_start.device) 
        z_t = self.q_sample(x_start, alpha_t, sigma_t, noise)

        model_pred = model(z_t, logsnr_t, **model_kwargs)
        if self.pred_term == ModelMeanType.EPSILON:
            eps_pred = model_pred
            target = noise
        elif self.pred_term == ModelMeanType.VELOCITY: # SID uses mse loss for velocity
            target = noise
            eps_pred = sigma_t * z_t + alpha_t * model_pred
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        
        # see https://arxiv.org/pdf/2303.09556
        #Convert to fp32 to precision
        eps_pred, target = eps_pred.float(), target.float()

        mse_target = (eps_pred - target) ** 2
        weight = self.get_weight(t)
        loss = mean_flat(weight * mse_target)
        terms = {
            'mse': mean_flat(mse_target),
            'loss': loss,
        }
        return terms
    def get_alpha_sigma_from_logsnr(self, logsnr_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(logsnr_t.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(logsnr_t.device)
        return alpha_t, sigma_t
    def x0_from_pred(self, x_t: torch.Tensor, t: torch.Tensor, model_pred: torch.Tensor)-> torch.Tensor:
        logsnr_t = self.logsnr_t(t, self.schedule).to(x_t.device)
        alpha_t, sigma_t = self.get_alpha_sigma_from_logsnr(logsnr_t)
        if self.pred_term == ModelMeanType.EPSILON:
            x_pred = (x_t - sigma_t * model_pred) / alpha_t
        elif self.pred_term == ModelMeanType.VELOCITY:
            x_pred = alpha_t * x_t - sigma_t * model_pred
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        return x_pred
    def get_timestep(self, x_start: torch.Tensor):
        """
        get a randomized timestep for each sample in the batch
        """
        t = torch.rand(x_start.shape[0], device=x_start.device)
        return t
    def get_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        get w^(λ_t)
        by default weighting, w^(λ_t) = p(λ_t) , in eps prediction loss
        the returned weight is divided by p(λ_t) so you can directly use get_weight(t) * (pred - eps) ** 2
        """
        lambda_t = self.logsnr_t(t, self.schedule).view(-1, 1, 1, 1).to(t.device)
        if self.loss_type == LossType.WEIGHTED_MSE:
            # do sigmoid weighting
            bias = - 2 * int(math.log2(1 / self.size_ratio))  + 1
            #print(f'bias: {bias}')
            sigmoid_weight_t = torch.sigmoid(-lambda_t + bias)
            # mse prediction weight is sech(lambda_t/2) = 2 / (exp(lambda_t/2) + exp(-lambda_t/2))
            mse_weight_t = 2 / (torch.exp(lambda_t / 2) + torch.exp(-lambda_t / 2))
            weight_t = sigmoid_weight_t / mse_weight_t 
            #weight_t = torch.ones_like(lambda_t)
            return weight_t
        elif self.loss_type == LossType.MSE:
            return torch.ones_like(lambda_t)
        else:
            raise NotImplementedError(f'Invalid loss type {self.loss_type}')
    def q_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, s:torch.Tensor, gamma: float = .3) -> tuple[torch.Tensor, torch.Tensor]:
        """
        gives q(x_s | x_start, x_t)
        x_start can be x_pred or actual x_0
        """
        logsnr_t = self.logsnr_t(t, self.schedule).to(x_t.device)
        logsnr_s = self.logsnr_t(s, self.schedule).to(x_t.device)
        alpha_t, sigma_t = self.get_alpha_sigma_from_logsnr(logsnr_t)
        alpha_s, sigma_s = self.get_alpha_sigma_from_logsnr(logsnr_s)
        alpha_ts = alpha_t / alpha_s
        sigma_st = sigma_s / sigma_t # <1, numerically accuracy higher than it's reciprocal in bf16
        assert torch.all(sigma_st < 1), f'sigma_st: {sigma_st}'
        sigma_ts_sq = 1 - alpha_ts ** 2
        mu = (sigma_st ** 2 * alpha_ts) * x_t + (sigma_ts_sq * alpha_s / sigma_t ** 2) * x_start
        max_var_log = torch.log(sigma_ts_sq)
        min_var_log = max_var_log + torch.log(sigma_st) * 2
        logvar = gamma * max_var_log + (1 - gamma) * min_var_log
        variance = torch.exp(logvar)
        return mu, variance
    def ddpm_sample(self, model, x_t, t: float, s:float, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0, gamma: float = .3):
        logsnr_t = self.logsnr_t(t, self.schedule).to(x_t.device)
        logsnr_s = self.logsnr_t(s, self.schedule).to(x_t.device)
        #print(f'logsnr_t: {logsnr_t}, logsnr_s: {logsnr_s}')
        model_pred = model(x_t, logsnr_t, **model_kwargs)
        x_pred = self.x0_from_pred(x_t, t, model_pred)
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)        
        mu, variance = self.q_mean_variance(x_pred, x_t, t, s, gamma)
        return mu, variance

    def ddim_sample_r(self, model, x_t, t: float, s:float, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0):
        logsnr_t = self.logsnr_t(t, self.schedule).to(x_t.device)
        logsnr_s = self.logsnr_t(s, self.schedule).to(x_t.device)
        #print(f'logsnr_t: {logsnr_t}, logsnr_s: {logsnr_s}')
        model_pred = model(x_t, logsnr_t, **model_kwargs)
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s)).view(-1, 1, 1, 1).to(x_t.device)
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s)).view(-1, 1, 1, 1).to(x_t.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x_t.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x_t.device)
        if self.pred_term == ModelMeanType.EPSILON:
            x_pred = (x_t - sigma_t * model_pred) / alpha_t
        else:
            raise NotImplementedError(f'Invalid pred_term {self.pred_term}')
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)        
        # for mu, variance, see https://arxiv.org/pdf/2410.19324
        #print(f'c: {c.shape}, alpha_t: {alpha_t.shape}, sigma_t: {sigma_t.shape}, model_pred: {model_pred.shape}')
        xt_ = alpha_s * x_pred + sigma_s * model_pred
        return xt_, torch.zeros_like(xt_)
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0, gamma: float = .3):
        if noise is None:
            noise = torch.randn(shape, device=device)
        x = noise.to(device)
        assert gamma != .3, f'Invalid gamma {gamma}'
        progess_bar = tqdm(reversed(range(1, len(self.used_timesteps)))) if progress else reversed(range(1, len(self.used_timesteps)))
        for i in progess_bar:
            #print(f'Processing timestep {i}:{self.used_timesteps[i]} to timestep {i - 1}:{self.used_timesteps[i - 1]}')
            u_t = self.used_timesteps[i] / self.diffusion_steps # current t
            u_s = self.used_timesteps[i - 1] / self.diffusion_steps # next t
            u_t = torch.tensor(u_t).to(device).repeat(x.size(0)) # repeat for batch size
            u_s = torch.tensor(u_s).to(device).repeat(x.size(0))
            z_mu, z_var = self.ddpm_sample(model, x, u_t, u_s, clip_denoised, denoised_fn, cond_fn, model_kwargs, eta, gamma)
            x = z_mu + torch.randn_like(z_mu) * torch.sqrt(z_var)
            xm.mark_step()
        t_lowest = self.used_timesteps[0] / self.diffusion_steps
        t_lowest = torch.tensor(t_lowest).to(device).repeat(x.size(0))
        logsnr_t_lowest = self.logsnr_t(t_lowest, self.schedule).to(device)
        model_pred = model(x, logsnr_t_lowest, **model_kwargs)
        x_pred = self.x0_from_pred(x, t_lowest, model_pred) # zero variance
        if clip_denoised:
            x_pred = x_pred.clamp(-1, 1)
        return x_pred
        