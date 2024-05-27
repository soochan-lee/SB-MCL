import torch.nn as nn
from einops import rearrange, pack


class MlpEncoder(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config['hidden_dim'], bias=False),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=False),
            nn.BatchNorm1d(config['hidden_dim']),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)
