import math
import torch
import torch.nn as nn

from models.maml_nn import MamlModule, MamlLinear


class MamlMlp(MamlModule):
    def __init__(self, config, enc_args):
        super().__init__(reptile=enc_args['reptile'])
        self.config = config
        self.enc_args = enc_args
        self.input_shape = tuple(enc_args['input_shape'])
        self.output_shape = tuple(enc_args['output_shape'])
        input_dim = math.prod(self.input_shape)
        output_dim = math.prod(self.output_shape)

        # Input layer
        layers = [
            MamlLinear(input_dim, enc_args['hidden_dim'], bias=True),
            nn.ReLU(inplace=True),
        ]

        # Additional layers
        for _ in range(enc_args['layers'] - 2):
            layers.append(MamlLinear(enc_args['hidden_dim'], enc_args['hidden_dim'], bias=True))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(MamlLinear(enc_args['hidden_dim'], output_dim, bias=True))
        if self.enc_args['output_activation'] == 'tanh':
            layers.append(nn.Tanh())
        elif self.enc_args['output_activation'] == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.enc_args['output_activation'] == 'none':
            pass
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if len(self.input_shape) > 1:
            x = torch.flatten(x, start_dim=-len(self.input_shape))
        assert len(x.shape) == 3, f'Expected shape for MamlMlp [batch, inner_batch, flat_dim], got {x.shape}'
        out = self.mlp(x)
        if len(self.output_shape) > 1:
            out = torch.unflatten(out, -1, self.output_shape)
        return out
