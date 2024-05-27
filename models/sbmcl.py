import math
import torch
import torch.nn as nn
from einops import rearrange, pack, reduce, repeat

from models import Model
from models.components import COMPONENT, Mlp, CnnEncoder
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN, kl_div


def sequential_bayes(mean, log_pre, prior_mean=None, prior_log_pre=None):
    # Aggregate
    log_pre_agg = torch.logsumexp(log_pre, dim=1, keepdim=True)
    mean_agg = reduce(mean * (log_pre - log_pre_agg).exp(), 'b l ... -> b 1 ...', 'sum')
    if prior_mean is None:
        return mean_agg, log_pre_agg

    post_log_pre = torch.logsumexp(torch.stack([log_pre_agg, prior_log_pre.expand_as(log_pre_agg)], dim=0), dim=0)
    post_mean = (
            mean_agg * (log_pre_agg - post_log_pre).exp() +
            prior_mean * (prior_log_pre - post_log_pre).exp()
    )
    return post_mean, post_log_pre


def slice_shots(data, tasks, shots):
    data = rearrange(data, 'b (t s) ... -> b t s ...', t=tasks)
    sliced = data[:, :, :shots]
    sliced = rearrange(sliced, 'b t s ... -> b (t s) ...')
    return sliced


class Sbmcl(Model):
    """Sequential Variational Bayes"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        x_enc_args = config['x_enc_args']
        x_enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['x_encoder']](config, x_enc_args)
        x_enc_shape = self.x_encoder.output_shape

        if 'xy_encoder' in self.config and self.config['xy_encoder'] == 'CompXYEncoder':
            self.y_encoder = None
            self.xy_encoder = CompXYEncoder(config)
        else:
            y_enc_args = config['y_enc_args']
            y_enc_args['input_shape'] = config['y_shape']
            self.y_encoder = COMPONENT[config['y_encoder']](config, y_enc_args)
            y_enc_shape = self.y_encoder.output_shape

            xy_enc_args = config['xy_enc_args']
            xy_enc_args['input_shape'] = (math.prod(x_enc_shape) + math.prod(y_enc_shape),)
            xy_enc_args['output_shape'] = (2, config['z_dim'])
            self.xy_encoder = Mlp(config, xy_enc_args)

        y_dec_args = config['y_dec_args']
        y_dec_args['output_shape'] = config['y_shape']
        self.y_decoder = COMPONENT[config['y_decoder']](config, y_dec_args)

        xz_enc_args = config['xz_enc_args']
        xz_enc_args['input_shape'] = (math.prod(x_enc_shape) + config['z_dim'],)
        xz_enc_args['output_shape'] = self.y_decoder.input_shape
        self.xz_encoder = Mlp(config, xz_enc_args)

        self.prior_mean = nn.Parameter(torch.zeros(config['z_dim']), requires_grad=True)
        self.prior_log_pre = nn.Parameter(torch.zeros(config['z_dim']), requires_grad=True)

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]

        #########
        # Train #
        #########
        start = 0
        chunk = self.config['train_chunk'] if 'train_chunk' in self.config else train_num
        post_mean = repeat(self.prior_mean, 'h -> b 1 h', b=batch)
        post_log_pre = repeat(self.prior_log_pre, 'h -> b 1 h', b=batch)
        while start < train_num:
            end = min(start + chunk, train_num)
            train_x_chunk = train_x[:, start:end]
            train_y_chunk = train_y[:, start:end]
            post_mean, post_log_pre = self.forward_train(train_x_chunk, train_y_chunk, post_mean, post_log_pre)
            start = end

        ########
        # Test #
        ########
        if 'train_recon' in self.config and self.config['train_recon'] and meta_split == 'train':
            tasks = self.config['tasks']
            train_ex_per_task = min(train_num, test_num) // tasks

            train_x_subset = slice_shots(train_x, tasks, train_ex_per_task)
            train_y_subset = slice_shots(train_y, tasks, train_ex_per_task)
            train_test_x = torch.cat([train_x_subset, test_x], dim=1)
            train_test_y = torch.cat([train_y_subset, test_y], dim=1)
        else:
            train_test_x = test_x
            train_test_y = test_y

        train_test_x_enc = self.x_encoder(train_test_x)
        train_test_x_enc = rearrange(train_test_x_enc, 'b l ... -> b l (...)')
        train_test_num = train_test_x.shape[1]

        if 'map' in self.config and self.config['map']:
            # MAP
            z = repeat(post_mean, 'b 1 h -> b l h', l=train_test_num)
            z_samples = 1
            test_xz, _ = pack([train_test_x_enc, z], 'b l *')
            test_xz_enc = self.xz_encoder(test_xz)
            train_test_y_hat = self.y_decoder(test_xz_enc)
        else:
            # Sample multiple z during meta-test
            z_samples = 1 if meta_split == 'train' else self.config['eval_z_samples']
            train_test_y_hat = None
            for _ in range(z_samples):
                # Sample z
                noise = torch.randn(batch, train_test_num, self.config['z_dim'], device=post_mean.device)
                z = post_mean + noise * (-post_log_pre / 2).exp()  # b l h

                # Decode
                test_xz, _ = pack([train_test_x_enc, z], 'b l *')
                test_xz_enc = self.xz_encoder(test_xz)
                decoder_out = self.y_decoder(test_xz_enc)
                if train_test_y_hat is None:
                    train_test_y_hat = decoder_out
                else:
                    train_test_y_hat += decoder_out
        if z_samples > 1:
            train_test_y_hat /= z_samples

        output = Output()
        meta_loss_train_test = self.loss_fn(train_test_y_hat, train_test_y)
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
            denom = (train_num + test_num) * math.prod(meta_loss_train_test.shape[2:])
            sbmcl_kl_loss = kl_div(post_mean, post_log_pre) / denom
            sbmcl_kl_loss = rearrange(sbmcl_kl_loss, 'b 1 -> b')
            if 'sbmcl_kl_weight' in self.config:
                sbmcl_kl_loss = sbmcl_kl_loss * self.config['sbmcl_kl_weight']
            meta_loss = meta_loss + sbmcl_kl_loss
            output[f'loss/sbmcl_kl/meta_{meta_split}'] = sbmcl_kl_loss

        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output

        if meta_split == 'train':
            output.add_sbmcl_prior_summary(self.prior_mean, self.prior_log_pre)

        if self.config['output_type'] == 'image':
            output.add_image_comparison_summary(
                test_x, test_y, test_x, train_test_y_hat, key=f'completion/meta_{meta_split}')
        return output

    def forward_train(self, train_x, train_y, prior_mean=None, prior_log_pre=None):
        if 'xy_encoder' in self.config and self.config['xy_encoder'] == 'CompXYEncoder':
            mean_log_pre = self.xy_encoder(train_x, train_y)
            mean_log_pre = rearrange(mean_log_pre, 'b l (z h) -> b l z h', z=2)
        else:
            # Encode x and y separately
            train_x_enc = self.x_encoder(train_x)
            train_y_enc = self.y_encoder(train_y)
            train_x_enc = rearrange(train_x_enc, 'b l ... -> b l (...)')
            train_y_enc = rearrange(train_y_enc, 'b l ... -> b l (...)')

            # Encode x and y together
            train_xy, _ = pack([train_x_enc, train_y_enc], 'b l *')
            mean_log_pre = self.xy_encoder(train_xy)

        mean, log_pre = torch.unbind(mean_log_pre, dim=-2)
        return sequential_bayes(mean, log_pre, prior_mean, prior_log_pre)


class CompXYEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_args = config['xy_enc_args']
        enc_args['input_shape'] = (
            config['x_shape'][0], config['x_shape'][1] + config['y_shape'][1], config['x_shape'][2])
        enc_args['output_shape'] = (2 * config['z_dim'],)
        self.cnn_encoder = CnnEncoder(config, enc_args=enc_args)

    def forward(self, train_x, train_y):
        train_xy, _ = pack([train_x, train_y], 'b l c * w')
        return self.cnn_encoder(train_xy)
