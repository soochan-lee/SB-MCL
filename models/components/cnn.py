import math
import torch.nn as nn
from einops import rearrange


class CnnEncoder(nn.Module):
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
            nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=1, padding=1, bias=False),
            # 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            # 4 x 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            # 2 x 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * h // 16 * w // 16, self.output_shape[0]),
        )

        if enc_args['output_activation'] == 'relu':
            self.net.add_module('last_bn', nn.BatchNorm1d(self.output_shape[0]))
            self.net.add_module('last_relu', nn.ReLU(inplace=True))
        elif enc_args['output_activation'] == 'none':
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        b, l = x.shape[:2]
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        out = self.net(x)
        return rearrange(out, '(b l) d -> b l d', b=b, l=l)


class CnnDecoder(nn.Module):
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
            nn.Linear(self.input_shape[0], math.prod(proj_shape), bias=False),
            nn.BatchNorm1d(math.prod(proj_shape)),
            nn.Unflatten(1, proj_shape),
            # 2 x 2
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 4 x 4
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 8 x 8
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 16 x 16
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 32 x 32
            nn.Conv2d(32, self.output_shape[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        if 'output_activation' not in self.dec_args:
            self.dec_args['output_activation'] = 'tanh'

        if self.dec_args['output_activation'] == 'tanh':
            self.net.add_module('output_activation', nn.Tanh())
        elif self.dec_args['output_activation'] == 'none':
            pass
        else:
            raise NotImplementedError

    def forward(self, x):
        b, l = x.shape[:2]
        x = rearrange(x, 'b l d -> (b l) d')
        out = self.net(x)
        return rearrange(out, '(b l) c h w -> b l c h w', b=b, l=l)
