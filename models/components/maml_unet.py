import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.components.unet import PositionalEmbedding, ResidualBlock, Upsample, Downsample
from models.maml_nn import MamlModule, MamlConv2d, MamlLinear


class MamlResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=F.relu):
        super().__init__()

        self.activation = activation

        self.conv_1 = MamlConv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            MamlConv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = MamlLinear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None
        self.residual_connection = MamlConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None, y=None):
        b, l = x.shape[:2]
        out = self.activation(x)
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            time_emb = rearrange(time_emb, '(b l) t -> b l t', b=b, l=l)
            out += rearrange(self.time_bias(self.activation(time_emb)), 'b l z -> b l z 1 1')

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(out)
        out = self.conv_2(out) + self.residual_connection(x)

        return out


class FastUNet(MamlModule):

    def __init__(
        self,
        img_channels,
        base_channels,
        channel_mults=(1, 2, 4, 8),
        num_bottleneck_layer=2,
        num_res_blocks=2,
        time_emb_dim=None,
        time_emb_scale=1.0,
        num_classes=None,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(),
        norm="gn",
        num_groups=32,
        initial_pad=0,
        reptile=False
    ):
        super().__init__(reptile=reptile)

        self.activation = activation
        self.initial_pad = initial_pad

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
    
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
        

        bottleneck_layer = []
        for _ in range(num_bottleneck_layer):
            bottleneck_layer.append(
                MamlResidualBlock(now_channels, now_channels, dropout, time_emb_dim=time_emb_dim,
                                  num_classes=num_classes, activation=activation)
            )
        self.mid = nn.ModuleList(bottleneck_layer)

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
            
            if i != 0:
                self.ups.append(Upsample(now_channels))
        
        assert len(channels) == 0
        
        self.out_norm = nn.GroupNorm(num_groups, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
    def forward(self, x, time=None, y=None):
        batch, inner_batch = x.shape[:2]
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            if isinstance(time, int):
                time = torch.tensor([time], device=x.device)
            elif len(time.shape) == 0:
                time = time[None]
            time = rearrange(time, '... -> (...)')
            time_emb = self.time_mlp(time)
            if time.numel() == 1:
                time_emb = repeat(time_emb, '1 h -> (b l) h', b=batch, l=inner_batch)
        else:
            time_emb = None
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
            
        # Fast weight
        x = rearrange(x, '(b l) c h w -> b l c h w', b=batch, l=inner_batch)
        for layer in self.mid:
            x = layer(x, time_emb, y)
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        if self.initial_pad != 0:
            x =x[:, :, ip:-ip, ip:-ip]
        
        return rearrange(x, '(b l) c h w -> b l c h w', b=batch, l=inner_batch)
