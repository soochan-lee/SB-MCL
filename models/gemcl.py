import torch
import torch.nn as nn
from einops import rearrange, reduce

from .components import COMPONENT
from .model import Model, Output


class GeMCL(Model):
    """Prototypical Network"""

    def __init__(self, config):
        super().__init__(config)
        x_enc_args = config['x_enc_args']
        x_enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['x_encoder']](config, x_enc_args)
        assert config['output_type'] == 'class'
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.map = config['map']
        self.alpha = nn.Parameter(torch.tensor(100.), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1000.), requires_grad=True)

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        batch, test_num = test_y.shape
        train_x_enc, test_x_enc = self.encode_x(train_x, test_x)

        # Train
        prototypes = reduce(
            train_x_enc, 'b (t s) d -> b t d', 'mean', t=self.config['tasks'], s=self.config['train_shots'])
        squared_diff = (
                rearrange(train_x_enc, 'b (t s) d -> b t s d', t=self.config['tasks']) -
                rearrange(prototypes, 'b t d -> b t 1 d')
        ).square()
        squared_diff = reduce(squared_diff, 'b t s d -> b t d', 'sum')

        alpha_prime = self.alpha + self.config['train_shots'] / 2
        beta_prime = self.beta + squared_diff / 2

        if self.map:
            var = beta_prime / (alpha_prime - 0.5)
        else:
            var = beta_prime / alpha_prime * (1 + 1 / self.config['train_shots'])

        # Test
        test_x_enc = rearrange(test_x_enc, 'b l h -> b l 1 h')
        prototypes = rearrange(prototypes, 'b t h -> b 1 t h')
        squared_diff = (test_x_enc - prototypes).square()  # b l t h
        var = rearrange(var, 'b t h -> b 1 t h')

        eps = 1e-8
        if self.map:
            nll = squared_diff / var + (var + eps).log()
            nll = reduce(nll, 'b l t h -> b l t', 'sum') / 2
        else:
            nll = (squared_diff / (var * alpha_prime * 2) + 1).log() * (alpha_prime + 0.5) + (var + eps).log()
            nll = reduce(nll, 'b l t h -> b l t', 'sum')

        logit = -nll
        loss = self.ce(rearrange(logit, 'b l t -> (b l) t'), rearrange(test_y, 'b l -> (b l)'))
        loss = reduce(loss, '(b n) -> b', 'mean', b=batch, n=test_num)

        output = Output()
        output[f'loss/meta_{meta_split}'] = loss
        if not summarize:
            return output

        output.add_classification_summary(logit, test_y, meta_split)
        return output
