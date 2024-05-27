import math
import torch
from torch import nn
from einops import rearrange, reduce

from models import Model
from models.components import COMPONENT
from models.model import Output


class ALPaCA(Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.loss_fn = nn.MSELoss(reduction='none')
        self.input_type = config['input_type']
        self.output_type = config['output_type']
        x_enc_args = config['x_enc_args']
        x_enc_args['input_shape'] = config['x_shape']
        x_enc_args['output_shape'] = (config['hidden_dim'],)
        self.x_encoder = COMPONENT[config['x_encoder']](config, x_enc_args)
        self.hidden_dim = math.prod(self.x_encoder.output_shape)
        self.y_dim = math.prod(config['y_shape'])

        # Initial mean
        self.mean0 = nn.Parameter(torch.zeros(self.hidden_dim, self.y_dim), requires_grad=True)

        # Cholesky decomposition of initial precision

        self.prec0_cholesky = nn.Parameter(torch.eye(self.hidden_dim), requires_grad=True)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        train_x_enc, test_x_enc = self.encode_x(train_x, test_x)

        # Flatten y
        test_y_shape = test_y.shape
        train_y = rearrange(train_y, 'b l ... -> b l (...)')
        test_y = rearrange(test_y, 'b l ... -> b l (...)')

        prec0_cholesky = torch.tril(self.prec0_cholesky)

        # Compute posterior covariance of the linear model weights
        prec0 = prec0_cholesky @ prec0_cholesky.T
        prec = torch.einsum('blh,blz->bhz', train_x_enc, train_x_enc) + prec0
        # cov = torch.linalg.inv(prec)

        # Compute posterior mean of the linear model weights
        xy = torch.einsum('blh,bly->bhy', train_x_enc, train_y) + (prec0 @ self.mean0)
        # mean = torch.einsum('bzh,bhy->bzy', cov, xy)
        mean = torch.linalg.solve(prec, xy)

        # Compute predictive mean
        pred_mean = torch.einsum('bzy,blz->bly', mean, test_x_enc)

        """
        If the evaluation measure is MSE, no need to train with the exact NLL
        
        # Partially compute predictive precision
        noise_prec = self.noise_prec_cholesky @ self.noise_prec_cholesky.T
        x_cov_x = torch.einsum('blh,bhz,blz->bl', test_x_enc, cov, test_x_enc)
        noise_scale = x_cov_x + 1
        # Computing the following first is inefficient
        # pred_prec = noise_prec / rearrange(noise_scale, 'b l -> b l 1 1')

        # Compute predictive NLL
        err = test_y - pred_mean
        err_prec_err = torch.einsum('bly,yw,blw->bl', err, noise_prec, err) / noise_scale

        pred_cov_log_det = self.y_dim * torch.log(noise_scale) - torch.logdet(noise_prec)
        pred_nll = pred_cov_log_det + err_prec_err

        loss = pred_nll
        """

        # MSE loss
        loss = reduce(self.loss_fn(pred_mean, test_y), 'b l y -> b', 'mean')
        output = Output()
        output[f'loss/meta_{meta_split}'] = loss

        return output
