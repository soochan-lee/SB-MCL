import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from .module import MamlModule


class MamlConv2d(MamlModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, reptile=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.param_inits = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(out_channels, in_channels, *kernel_size), requires_grad=not reptile)
        ])
        if bias:
            self.param_inits.append(nn.Parameter(torch.FloatTensor(out_channels), requires_grad=not reptile))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.params = None

        self.initialize()

    def initialize(self):
        nn.init.kaiming_uniform_(self.param_inits[0], a=math.sqrt(5))
        if len(self.param_inits) == 2:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.param_inits[0])
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.param_inits[1], -bound, bound)

    def forward(self, input):
        # Get fast params
        if self.params is None:
            raise RuntimeError('Fast weights are not initialized')
        if len(self.params) == 1:
            w = self.params[0]
            b = None
        elif len(self.params) == 2:
            w, b = self.params
        else:
            raise RuntimeError('Invalid fast weights')

        w = rearrange(w, 'b o i h w -> (b o) i h w')
        if b is not None:
            b = rearrange(b, 'b o -> (b o)')

        assert len(input.shape) == 5, 'Input shape must be [batch, inner_batch, channels, height, width]'
        batch = input.shape[0]
        input = rearrange(input, 'b n i h w -> n (b i) h w')
        output = F.conv2d(input, weight=w, bias=b, stride=self.stride, padding=self.padding, dilation=self.dilation,
                          groups=batch)
        output = rearrange(output, 'n (b o) h w -> b n o h w', b=batch)

        return output
