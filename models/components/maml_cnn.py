import math
import torch.nn as nn
from einops import rearrange

from models.maml_nn import MamlModule, MamlLinear, MamlConv2d


class MamlCnnEncoder(MamlModule):
    def __init__(self, config, enc_args):
        super().__init__()
        self.config = config
        self.enc_args = enc_args
        self.input_shape = enc_args['input_shape']
        self.output_shape = enc_args['output_shape']
        assert len(self.input_shape) == 3
        assert len(self.output_shape) == 1

        c, h, w = self.input_shape
        self.net = nn.Sequential(
            # 32 x 32
            MamlConv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1, bias=True),
            # 32 x 32
            nn.ReLU(inplace=True),
            MamlConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            # 16 x 16
            nn.ReLU(inplace=True),
            MamlConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            # 8 x 8
            nn.ReLU(inplace=True),
            MamlConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            # 4 x 4
            nn.ReLU(inplace=True),
            MamlConv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            # 2 x 2
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            MamlLinear(256 * h // 16 * w // 16, self.output_shape[0]),
        )

        if enc_args['output_activation'] == 'relu':
            self.net.add_module('last_relu', nn.ReLU(inplace=True))
        elif enc_args['output_activation'] == 'none':
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.net(x)


class MamlCnnDecoder(MamlModule):
    def __init__(self, config, dec_args):
        super().__init__()
        self.config = config
        self.dec_args = dec_args
        self.input_shape = dec_args['input_shape']
        self.output_shape = dec_args['output_shape']
        assert len(self.input_shape) == 1
        assert len(self.output_shape) == 3
        c, h, w = self.output_shape
        proj_shape = [256, h // 16, w // 16]
        self.net = nn.Sequential(
            MamlLinear(self.input_shape[0], math.prod(proj_shape), bias=True),
            nn.Unflatten(2, proj_shape),
            # 2 x 2
            nn.ReLU(inplace=True),
            MamlUpsample(scale_factor=2, mode='nearest'),
            # 4 x 4
            MamlConv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            MamlUpsample(scale_factor=2, mode='nearest'),
            # 8 x 8
            MamlConv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            MamlUpsample(scale_factor=2, mode='nearest'),
            # 16 x 16
            MamlConv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            MamlUpsample(scale_factor=2, mode='nearest'),
            # 32 x 32
            MamlConv2d(32, self.output_shape[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class MamlUpsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        assert len(x.shape) == 5, \
            f'Expected shape for MamlUpsample [batch, inner_batch, channels, height, width], got {x.shape}'
        b, n, c, h, w = x.shape
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.upsample(x)
        x = rearrange(x, '(b n) c h w -> b n c h w', b=b, n=n)
        return x
