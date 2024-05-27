import math
import torch
import torch.nn as nn
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, config, enc_args):
        super().__init__()
        self.config = config
        self.enc_args = enc_args
        self.input_shape = tuple(enc_args['input_shape'])
        self.output_shape = tuple(enc_args['output_shape'])
        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)

        # Input layer
        layers = [
            nn.Linear(input_dim, enc_args['hidden_dim'], bias=False),
            nn.BatchNorm1d(enc_args['hidden_dim']),
            nn.ReLU(inplace=True),
        ]

        # Additional layers
        for _ in range(enc_args['layers'] - 2):
            layers.append(nn.Linear(enc_args['hidden_dim'], enc_args['hidden_dim'], bias=False))
            layers.append(nn.BatchNorm1d(enc_args['hidden_dim']))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Linear(enc_args['hidden_dim'], output_dim, bias=True))
        if self.enc_args['output_activation'] == 'tanh':
            layers.append(nn.Tanh())
        elif self.enc_args['output_activation'] == 'relu':
            layers.append(nn.BatchNorm1d(enc_args['hidden_dim']))
            layers.append(nn.ReLU(inplace=True))
        elif self.enc_args['output_activation'] == 'none':
            pass
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        b, l = x.shape[:2]
        x = rearrange(x, 'b l ... -> (b l) (...)')
        out = self.mlp(x)
        out = torch.unflatten(out, -1, self.output_shape)
        out = rearrange(out, '(b l) ... -> b l ...', b=b, l=l)
        return out
