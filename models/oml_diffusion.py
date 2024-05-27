import math

import torch
from einops import rearrange, reduce
from torch import nn

from models.components import COMPONENT
from models.maml_nn import MamlModule
from models.components.diffusion import Diffusion
from models.model import Output


class OmlDDPM(Diffusion, MamlModule):
    def __init__(self, config):
        Diffusion.__init__(self, config)
        MamlModule.__init__(self, reptile=config['reptile'])

        self.log_inner_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])
        self.input_type = config['input_type']
        self.output_type = config['output_type']

        self.set_denoiser(config)
        self.loss_fn = nn.MSELoss(reduction='none')

        # Build beta schedule
        diffusion_args = config['diffusion_args']
        beta_min = diffusion_args['beta_min']
        beta_max = diffusion_args['beta_max']
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


    def set_denoiser(self, config):
        # UNet
        denoiser_args = config['unet_args']
        denoiser_args['img_channels'] = config['x_shape'][0]
        denoiser_args['reptile'] = config['reptile']

        self.denoiser = COMPONENT[config['backbone']](**denoiser_args)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split, generation=False):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]
        is_meta_training = meta_split == 'train'
        train_t = torch.randint(low=0, high=self.n_times, size=(batch, train_num), device=train_x.device)
        test_t = torch.randint(low=0, high=self.n_times, size=(batch, test_num), device=test_x.device)

        # Forward process
        noisy_train_x, train_noise = self.make_noisy(train_x, train_t)
        noisy_test_x, test_noise = self.make_noisy(test_x, test_t)

        inner_lr = self.log_inner_lr.exp()

        self.denoiser.train()
        with torch.enable_grad():
            self.reset_fast_params(batch)

            # Inner loop
            for i in range(train_num):
                # Sequentially forward training data
                pred_noise_train = self.denoiser(noisy_train_x[:, i:i + 1], train_t[:, i:i + 1])
                loss = reduce(self.loss_fn(pred_noise_train, train_noise[:, i:i + 1]), 'b 1 ... -> b', 'mean').sum()
                self.inner_update(loss, inner_lr, is_meta_training=is_meta_training)

        # Forward test data
        pred_noise_test = self.denoiser(noisy_test_x, test_t)
        meta_loss = reduce(self.loss_fn(pred_noise_test, test_noise), 'b l c h w -> b', 'mean')

        if self.reptile and is_meta_training:
            self.reptile_update(self.config['reptile_lr'])

        output = Output()
        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output

        if meta_split == 'train':
            output['lr_inner'] = rearrange(inner_lr.detach(), '-> 1')

            if generation:
                generated_images = self.generate(batch)
                generated_images = rearrange(generated_images, 'b 1 c h w -> 1 b c h w')
                output['generation/raw'] = (generated_images + 1.) / 2
                test_x_images = rearrange(test_x[:, 0:1, :], 'b 1 c h w -> 1 b c h w')
                output.add_image_comparison_summary(test_x_images, generated_images,
                                                    key=f'generation/meta_{meta_split}')

        return output

    def generate(self, batch_size):
        x_t = torch.randn(batch_size, 1, self.img_C, self.img_H, self.img_W).to(self.betas.device)
        # Sampling
        with torch.no_grad():
            for t in range(self.n_times - 1, -1, -1):
                epsilon_pred = self.denoiser(x_t, t)
                x_t = self.denoise_at_t(x_t, t, epsilon_pred)
            x_t = x_t.clamp(-1., 1.)
        return x_t
