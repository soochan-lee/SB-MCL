from abc import *

import torch
import torch.nn as nn
from einops import rearrange


class Diffusion(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config
        diffusion_args = config['diffusion_args']
        image_resolution = config['x_shape']

        beta_min = diffusion_args['beta_min']
        beta_max = diffusion_args['beta_max']
        self.n_times = diffusion_args['n_times']

        self.img_C, self.img_H, self.img_W = image_resolution
        self.loss_fn = nn.MSELoss(reduction='none')

        # Build beta schedule
        beta_1, beta_T = beta_min, beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=self.n_times)
        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_betas', torch.sqrt(betas))

        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sqrt_alphas', torch.sqrt(self.alphas))
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1 - alpha_bars))
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('nll_coefficient', 0.5 * betas / (self.alphas * (1. - alpha_bars)))

    @abstractmethod
    def set_denoiser(self, config):
        pass

    def extract(self, a, t, x_shape):
        b, l, *_ = x_shape
        t = rearrange(t, 'b l -> (b l)')
        out = a.gather(-1, t)
        return out.reshape(b, l, *((1,) * (len(x_shape) - 2)))

    def make_noisy(self, x_zeros, t):
        epsilon = torch.randn_like(x_zeros).to(self.betas.device)

        if isinstance(t, torch.Tensor):
            sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
            sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        elif isinstance(t, int):
            sqrt_alpha_bar = self.sqrt_alpha_bars[t]
            sqrt_alpha_bar = rearrange(sqrt_alpha_bar, 'b l -> b l 1 1 1')
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
            sqrt_one_minus_alpha_bar = rearrange(sqrt_one_minus_alpha_bar, 'b l -> b l 1 1 1')
        else:
            raise NotImplementedError
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample.detach(), epsilon

    def denoise_at_t(self, x_t, t, epsilon_pred):
        if t > 1:
            noise = torch.randn_like(x_t).to(self.betas.device)
        else:
            noise = torch.zeros_like(x_t).to(self.betas.device)

        alpha = self.alphas[t]
        sqrt_alpha = self.sqrt_alphas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
        sqrt_beta = self.sqrt_betas[t]

        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1. - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) \
                      + sqrt_beta * noise
        return x_t_minus_1

    def generate(self, num_samples, z=None):
        x_t = torch.randn(1, num_samples, self.img_C, self.img_H, self.img_W).to(self.betas.device)
        # Sampling
        with torch.no_grad():
            for t in range(self.n_times - 1, -1, -1):
                epsilon_pred = self.denoiser(x_t, t, z=z)
                x_t = self.denoise_at_t(x_t, t, epsilon_pred)
            x_t = x_t.clamp(-1., 1.)
        return x_t