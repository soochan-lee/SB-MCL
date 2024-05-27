import torch
import torch.nn as nn
from einops import repeat, rearrange

from .module import MamlModule


class MamlLinear(MamlModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, reptile: bool = False):
        super().__init__()
        self.param_inits = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=not reptile)
        ])
        if bias:
            self.param_inits.append(nn.Parameter(torch.FloatTensor(out_features), requires_grad=not reptile))

        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.param_inits[0])
        if len(self.param_inits) == 2:
            nn.init.zeros_(self.param_inits[1])

    def forward(self, input):
        # Get fast params
        if len(self.params) == 1:
            w = self.params[0]
            b = None
        elif len(self.params) == 2:
            w, b = self.params
        else:
            raise RuntimeError('Invalid fast weights')

        # Forward
        assert len(input.shape) == 3, 'Input shape must be [batch, inner_batch, dim]'
        output = torch.einsum('boi,bni->bno', w, input)
        if b is not None:
            output = output + rearrange(b, 'b o -> b 1 o')

        return output
