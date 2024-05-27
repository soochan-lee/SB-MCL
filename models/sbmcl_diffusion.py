import math
import torch
import torch.nn as nn
from einops import reduce, repeat, rearrange

from models.components import COMPONENT
from models.components import Mlp
from models.components.diffusion import Diffusion
from models.model import Output
from models.sbmcl import sequential_bayes, slice_shots
from utils import kl_div


class SbmclDDPM(Diffusion):
    def __init__(self, config):
        super().__init__(config)

        self.input_type = config['input_type']
        self.output_type = config['output_type']

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['x_encoder']](config, enc_args)
        self.x_dims = math.prod(config['x_shape'])

        sbmcl_mlp_args = config['sbmcl_mlp_args']
        unet_args = config['unet_args']
        hidden_dim = unet_args['base_channels'] * unet_args['channel_mults'][-1]
        sbmcl_mlp_args['input_shape'] = self.encoder.output_shape
        self.z_shape = [hidden_dim, 4, 4]
        sbmcl_mlp_args['output_shape'] = [2] + self.z_shape
        self.latent_shape = sbmcl_mlp_args['input_shape']
        self.sbmcl_mlp = Mlp(config, sbmcl_mlp_args)

        self.set_denoiser(config)

        self.prior_mean = nn.Parameter(torch.zeros(self.z_shape), requires_grad=True)
        self.prior_log_pre = nn.Parameter(torch.zeros(self.z_shape), requires_grad=True)

    def set_denoiser(self, config):
        # UNet
        unet_args = config['unet_args']
        unet_args['img_channels'] = config['x_shape'][0]
        self.denoiser = COMPONENT[config['backbone']](**unet_args)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split, generation=False):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]

        #########
        # Train #
        #########
        train_x_enc = self.encoder(train_x)
        dist = self.sbmcl_mlp(train_x_enc)
        mean, log_pre = torch.unbind(dist, dim=2)

        # Aggregate
        post_mean, post_log_pre = sequential_bayes(mean, log_pre, self.prior_mean, self.prior_log_pre)

        ########
        # Test #
        ########
        if 'train_recon' in self.config and self.config['train_recon'] and meta_split == 'train':
            tasks = self.config['tasks']
            train_ex_per_task = min(train_num, test_num) // tasks

            train_x_subset = slice_shots(train_x, tasks, train_ex_per_task)
            train_test_x = torch.cat([train_x_subset, test_x], dim=1)
        else:
            train_test_x = test_x

        # Forward process
        train_test_num = train_test_x.shape[1]
        time = torch.randint(low=0, high=self.n_times, size=(batch, train_test_num), device=train_x.device)
        train_test_x_t, train_test_noise = self.make_noisy(train_test_x, time)

        if 'map' in self.config and self.config['map']:
            z = repeat(post_mean, 'b 1 ... -> b l ...', l=train_test_num)
            pred_noise = self.denoiser(train_test_x_t, time, z=z)
            meta_loss_train_test = reduce(self.loss_fn(pred_noise, train_test_noise), 'b l c h w -> b l', 'mean')
        else:
            # Sample multiple z during meta-test
            z_samples = 1 if meta_split == 'train' else self.config['eval_z_samples']
            meta_loss_sum = None
            for _ in range(z_samples):
                # Sample z
                noise = torch.randn(batch, train_test_num, *self.z_shape, device=post_mean.device)
                z = post_mean + noise * (-post_log_pre / 2).exp()  # b l ...
                pred_noise = self.denoiser(train_test_x_t, time, z=z)
                meta_loss_train_test = reduce(self.loss_fn(pred_noise, train_test_noise), 'b l c h w -> b l', 'mean')
                if meta_loss_sum is None:
                    meta_loss_sum = meta_loss_train_test
                else:
                    meta_loss_sum = meta_loss_sum + meta_loss_train_test
            meta_loss_train_test = meta_loss_sum / z_samples

        output = Output()
        if test_num < train_test_num:
            assert meta_split == 'train'
            meta_loss_train = reduce(meta_loss_train_test[:, :-test_num], 'b l -> b', 'mean')
            meta_loss_test = reduce(meta_loss_train_test[:, -test_num:], 'b l -> b', 'mean')
            train_weight = train_num / (train_num + test_num)
            test_weight = test_num / (train_num + test_num)
            meta_loss = train_weight * meta_loss_train + test_weight * meta_loss_test
            output[f'loss/meta_{meta_split}/train'] = meta_loss_train
            output[f'loss/meta_{meta_split}/test'] = meta_loss_test
        else:
            meta_loss = reduce(meta_loss_train_test, 'b l -> b', 'mean')

        if 'sbmcl_kl' in self.config and self.config['sbmcl_kl'] and meta_split == 'train':
            sbmcl_kl_loss = kl_div(post_mean, post_log_pre) / ((train_num + test_num) * self.x_dims)
            sbmcl_kl_loss = rearrange(sbmcl_kl_loss, 'b 1 -> b')
            if 'sbmcl_kl_weight' in self.config:
                sbmcl_kl_loss = sbmcl_kl_loss * self.config['sbmcl_kl_weight']
            meta_loss = meta_loss + sbmcl_kl_loss
            output[f'loss/sbmcl_kl/meta_{meta_split}'] = sbmcl_kl_loss

        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output
        if meta_split == 'train':
            # Evaluation
            if generation:
                generated_images = self.generate(8, z[:1, :1])
                output['generation/raw'] = (self.generate(test_num, z[:1, :1]) + 1.) / 2
                output.add_image_comparison_summary(test_x[0:1, 0:8, :], generated_images, key=f'generation/meta_{meta_split}')

        return output
