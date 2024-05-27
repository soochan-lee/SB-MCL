import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from models import Model
from models.components import COMPONENT, MamlMlp
from models.maml_nn import MamlModule
from models.model import Output
from utils import binarize, kl_div, nll_to_bpd, reparameterize


class OmlVae(Model, MamlModule):
    def __init__(self, config):
        Model.__init__(self, config)
        MamlModule.__init__(self, reptile=config['reptile'])
        self.log_inner_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['encoder']](config, enc_args)
        self.x_dims = math.prod(config['x_shape'])

        dec_args = config['dec_args']
        dec_args['output_shape'] = config['x_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        enc_mlp_args = config['enc_mlp_args']
        enc_mlp_args['input_shape'] = self.x_encoder.output_shape
        enc_mlp_args['output_shape'] = (2, config['latent_dim'])
        enc_mlp_args['reptile'] = config['reptile']
        self.enc_mlp = MamlMlp(config, enc_mlp_args)

        dec_mlp_args = config['dec_mlp_args']
        dec_mlp_args['input_shape'] = [config['latent_dim']]
        dec_mlp_args['output_shape'] = self.decoder.input_shape
        dec_mlp_args['reptile'] = config['reptile']
        self.dec_mlp = MamlMlp(config, dec_mlp_args)

        self.kl_weight = 0.0

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split, generation=False):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]
        is_meta_training = meta_split == 'train'

        train_x = binarize(train_x)
        test_x = binarize(test_x)
        train_x_enc, test_x_enc = self.encode_x(2 * train_x - 1, 2 * test_x - 1)

        inner_lr = self.log_inner_lr.exp()
        self.kl_weight += 1. / self.config['kl_warmup']
        self.kl_weight = min(self.kl_weight, 1.0)
        kl_weight = self.kl_weight if meta_split == 'train' else 1.0

        with torch.enable_grad():
            self.reset_fast_params(batch)

            # Inner loop
            for i in range(train_num):
                # Sequentially forward training data
                x_enc_i = train_x_enc[:, i:i + 1]
                latent_mean_log_var = self.enc_mlp(x_enc_i)
                latent_mean, latent_log_var = torch.unbind(latent_mean_log_var, dim=-2)
                latent = reparameterize(latent_mean, latent_log_var)
                logit = self.decode(latent)
                recon_loss = F.binary_cross_entropy_with_logits(logit, train_x[:, i:i + 1], reduction='none').sum()
                kl_loss = kl_div(latent_mean, latent_log_var).sum()
                loss = nll_to_bpd(recon_loss + kl_loss * kl_weight, self.x_dims)
                self.inner_update(loss, inner_lr, is_meta_training=is_meta_training)

        # Forward test data
        output = Output()
        if generation:
            latent = torch.randn(batch, test_num, self.config['latent_dim'], device='cuda') # b l h
            logit = self.decoder(latent)
            output['generation/raw'] = torch.sigmoid(logit)
            output.add_image_comparison_summary(test_x, torch.sigmoid(logit), key=f'generation/meta_{meta_split}')

        latent_mean_log_var = self.enc_mlp(test_x_enc)
        latent_mean, latent_log_var = torch.unbind(latent_mean_log_var, dim=-2)
        kl_loss = kl_div(latent_mean, latent_log_var)

        latent_samples = self.config['eval_latent_samples'] if meta_split == 'test' else 1
        recon_loss = torch.zeros_like(kl_loss)
        for _ in range(latent_samples):
            latent = reparameterize(latent_mean, latent_log_var)
            logit = self.decode(latent)
            bce = F.binary_cross_entropy_with_logits(logit, test_x, reduction='none')
            bce = reduce(bce, 'b l c h w -> b l', 'sum')
            recon_loss = recon_loss + bce
        recon_loss = recon_loss / latent_samples

        meta_loss = nll_to_bpd(recon_loss + kl_loss * kl_weight, self.x_dims)
        meta_loss = reduce(meta_loss, 'b l -> b', 'mean')

        if self.reptile and is_meta_training:
            self.reptile_update(self.config['reptile_lr'])

        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output

        if meta_split == 'train':
            output['lr_inner'] = rearrange(inner_lr.detach(), '-> 1')

        output[f'loss/kl/meta_{meta_split}'] = reduce(kl_loss, 'b l -> b', 'mean')
        output[f'loss/recon/meta_{meta_split}'] = reduce(recon_loss, 'b l -> b', 'mean')
        output.add_image_comparison_summary(test_x, torch.sigmoid(logit), key=f'recon/meta_{meta_split}')
        output['recon/raw'] = torch.sigmoid(logit)
        output['original/raw'] = test_x
        return output

    def decode(self, latent):
        dec_in = self.dec_mlp(latent)
        return self.decoder(dec_in)
