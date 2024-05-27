import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, reduce, repeat, rearrange

from models import Model
from models.components import COMPONENT, Mlp
from models.model import Output
from models.sbmcl import sequential_bayes, slice_shots
from utils import binarize, reparameterize, kl_div, nll_to_bpd


class SbmclVae(Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['encoder']](config, enc_args)
        self.x_dims = math.prod(config['x_shape'])

        dec_args = config['dec_args']
        dec_args['input_shape'] = [config['z_dim'] + config['latent_dim']]
        dec_args['output_shape'] = config['x_shape']
        dec_args['output_activation'] = 'none'
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        sbmcl_mlp_args = config['sbmcl_mlp_args']
        sbmcl_mlp_args['input_shape'] = self.x_encoder.output_shape
        sbmcl_mlp_args['output_shape'] = [2, config['z_dim']]
        self.sbmcl_mlp = Mlp(config, sbmcl_mlp_args)

        vae_mlp_args = config['vae_mlp_args']
        vae_mlp_args['input_shape'] = [config['z_dim'] + config['latent_dim']]
        vae_mlp_args['output_shape'] = [2, config['latent_dim']]
        self.vae_mlp = Mlp(config, vae_mlp_args)

        self.register_buffer('kl_weight', torch.zeros([]))
        self.prior_mean = nn.Parameter(torch.zeros(config['z_dim']), requires_grad=True)
        self.prior_log_pre = nn.Parameter(torch.zeros(config['z_dim']), requires_grad=True)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split, generation=False):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]
        train_x = binarize(train_x)
        test_x = binarize(test_x)

        # Encode x
        train_x_enc, test_x_enc = self.encode_x(2 * train_x - 1, 2 * test_x - 1)

        dist = self.sbmcl_mlp(train_x_enc)
        mean, log_pre = torch.unbind(dist, dim=-2)
        post_mean, post_log_pre = sequential_bayes(mean, log_pre, self.prior_mean, self.prior_log_pre)

        ########
        # Test #
        ########
        if 'train_recon' in self.config and self.config['train_recon'] and meta_split == 'train':
            tasks = self.config['tasks']
            train_ex_per_task = min(train_num, test_num) // tasks

            train_x_subset = slice_shots(train_x, tasks, train_ex_per_task)
            train_x_enc_subset = slice_shots(train_x_enc, tasks, train_ex_per_task)
            train_test_x = torch.cat([train_x_subset, test_x], dim=1)
            train_test_x_enc = torch.cat([train_x_enc_subset, test_x_enc], dim=1)
        else:
            train_test_x = test_x
            train_test_x_enc = test_x_enc
        train_test_num = train_test_x_enc.shape[1]

        self.kl_weight += 1. / self.config['kl_warmup']
        self.kl_weight.clamp_(max=1.0)

        if 'map' in self.config and self.config['map']:
            z = repeat(post_mean, 'b 1 h -> b l h', l=train_test_num)
            meta_loss_train_test, recon_loss, kl_loss, logit = self.vae_loss(
                train_test_x, train_test_x_enc, z, meta_split)
        else:
            # Sample multiple z during meta-test
            z_samples = 1 if meta_split == 'train' else self.config['eval_z_samples']
            meta_loss_train_test_sum = None
            recon_loss_sum = None
            kl_loss_sum = None
            for _ in range(z_samples):
                # Sample z
                noise = torch.randn(batch, train_test_num, self.config['z_dim'], device=mean.device)
                z = post_mean + noise * (-post_log_pre * 0.5).exp()  # b 1 h

                meta_loss_train_test, recon_loss, kl_loss, logit = self.vae_loss(
                    train_test_x, train_test_x_enc, z, meta_split)
                if meta_loss_train_test_sum is None:
                    meta_loss_train_test_sum = meta_loss_train_test
                    recon_loss_sum = recon_loss
                    kl_loss_sum = kl_loss
                else:
                    meta_loss_train_test_sum = meta_loss_train_test_sum + meta_loss_train_test
                    recon_loss_sum = recon_loss_sum + recon_loss
                    kl_loss_sum = kl_loss_sum + kl_loss

            if z_samples > 1:
                meta_loss_train_test = meta_loss_train_test_sum / z_samples
                recon_loss = recon_loss_sum / z_samples
                kl_loss = kl_loss_sum / z_samples

        output = Output()
        if generation:
            latent = torch.randn_like(z) # b l h
            z_latent = torch.cat([z, latent], dim=-1)
            logit = self.decoder(z_latent)
            output.add_image_comparison_summary(test_x, torch.sigmoid(logit), key=f'generation/meta_{meta_split}')
            output['generation/raw'] = torch.sigmoid(logit)
            return output

        if test_num < train_test_num:
            assert meta_split == 'train'
            meta_loss_train = reduce(meta_loss_train_test[:, :-test_num], 'b l ... -> b', 'mean')
            meta_loss_test = reduce(meta_loss_train_test[:, -test_num:], 'b l ... -> b', 'mean')
            train_weight = train_num / (train_num + test_num)
            test_weight = test_num / (train_num + test_num)
            meta_loss = train_weight * meta_loss_train + test_weight * meta_loss_test
            output[f'loss/meta_{meta_split}/train'] = meta_loss_train
            output[f'loss/meta_{meta_split}/test'] = meta_loss_test
        else:
            meta_loss = reduce(meta_loss_train_test, 'b l ... -> b', 'mean')

        if 'sbmcl_kl' in self.config and self.config['sbmcl_kl'] and meta_split == 'train':
            sbmcl_kl_loss = kl_div(post_mean, post_log_pre) / (train_num + test_num)
            sbmcl_kl_loss = nll_to_bpd(sbmcl_kl_loss, self.x_dims)
            sbmcl_kl_loss = rearrange(sbmcl_kl_loss, 'b 1 -> b')
            meta_loss = meta_loss + sbmcl_kl_loss
            output[f'loss/sbmcl_kl/meta_{meta_split}'] = sbmcl_kl_loss

        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output

        if meta_split == 'train':
            output.add_sbmcl_prior_summary(self.prior_mean, self.prior_log_pre)

        output[f'loss/kl/meta_{meta_split}'] = reduce(kl_loss, 'b l -> b', 'mean')
        output[f'loss/recon/meta_{meta_split}'] = reduce(recon_loss, 'b l -> b', 'mean')
        output.add_image_comparison_summary(test_x, torch.sigmoid(logit), key=f'recon/meta_{meta_split}')
        output['recon/raw'] = torch.sigmoid(logit)
        output['original/raw'] = test_x

        return output

    def vae_loss(self, x, x_enc, z, meta_split):
        train_test_xz_enc, _ = pack([x_enc, z], 'b l *')
        dist = self.vae_mlp(train_test_xz_enc)
        latent_mean, latent_log_var = torch.unbind(dist, dim=-2)
        kl_loss = kl_div(latent_mean, latent_log_var)
        latent_samples = self.config['eval_latent_samples'] if meta_split == 'test' else 1
        recon_loss = None
        for _ in range(latent_samples):
            latent = reparameterize(latent_mean, latent_log_var)
            z_latent = torch.cat([z, latent], dim=-1)
            logit = self.decoder(z_latent)
            bce = F.binary_cross_entropy_with_logits(logit, x, reduction='none')
            bce = reduce(bce, 'b l c h w -> b l', 'sum')
            if recon_loss is None:
                recon_loss = bce
            else:
                recon_loss = recon_loss + bce
        recon_loss = recon_loss / latent_samples
        kl_weight = self.kl_weight if meta_split == 'meta_train' else 1.0
        vae_loss = nll_to_bpd(recon_loss + kl_loss * kl_weight, self.x_dims)
        return vae_loss, recon_loss, kl_loss, logit
