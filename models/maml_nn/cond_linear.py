import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .module import MamlModule


class MamlCondLinear(MamlModule):
    def __init__(self, in_features: int, out_features: int, num_embed: int, bias: bool = True, reptile: bool = False):
        super().__init__(reptile)
        self.param_inits = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=not reptile)
        ])
        if bias:
            self.param_inits.append(nn.Parameter(torch.FloatTensor(out_features), requires_grad=not reptile))

        self.param_inits.append(nn.Parameter(torch.FloatTensor(num_embed, out_features), requires_grad=not reptile))

    def initialize(self):
        nn.init.xavier_normal_(self.param_inits[0])
        if len(self.param_inits) == 2:
            nn.init.zeros_(self.param_inits[1])
        if len(self.param_inits) == 3:
            nn.init.normal_(self.param_inits[2])

    def forward(self, input, cond):
        # Get fast params
        if len(self.params) == 2:
            w, embed = self.params
            b = None
        elif len(self.params) == 3:
            w, b, embed = self.params
        else:
            raise RuntimeError('Invalid fast weights')

        # Forward
        assert len(input.shape) == 3, 'Input shape must be [batch, inner_batch, dim]'
        assert len(cond.shape) == 2, 'Cond shape must be [batch, inner_batch]'
        output = torch.einsum('boi,bni->bno', w, input)
        if b is not None:
            output = output + rearrange(b, 'b o -> b 1 o')
        cond_output = torch.stack([F.embedding(cond[i], embed[i]) for i in range(input.shape[0])], dim=0) * output

        return cond_output
