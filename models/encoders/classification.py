import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from torch import Tensor


class ClassEncoder(nn.Module):
    def __init__(self, config, use_embedding=True):
        super().__init__()
        self.config = config
        self.embed_dim = config['hidden_dim']
        self.embedding = None
        if use_embedding:
            self.embedding = nn.Embedding(config['y_vocab'], self.embed_dim)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def encode(self, y: Tensor) -> Tensor:
        """Encode integer labels to vectors

        Args:
            y: [seq_len, batch, y_len]

        Returns:
            [seq_len, batch, y_len, hidden]
        """
        return self.embedding(y)

    def loss(self, logit: Tensor, y_code: Tensor) -> Tensor:
        """Compute loss

        Args:
            logit: [*, y_vocab]
            y_code: [*]

        Returns:
            [*]
        """
        return self.cross_entropy(logit.reshape(-1, logit.shape[-1]), y_code.reshape(-1)).view(*y_code.shape)

    def evaluate(self, logit: Tensor, y_code=None, y=None, y_codebook=None, exact_only=True):
        """Evaluate

        Args:
            logit: [batch, seq_len, y_len, y_vocab]
            y_code: [batch, seq_len, y_len]
            y: [batch, seq_len]
            y_codebook: [batch, num_cls, y_len]
            exact_only: if True, only count exact matches as correct
        Returns:
            [seq_len, batch]
        """

        if exact_only:
            assert y_code is not None
            pred = logit.argmax(-1)  # [batch, seq_len, y_len]
            return (pred == y_code).all(dim=2)  # [batch, seq_len]
        else:
            assert y is not None and y_codebook is not None
            log_sm = torch.log_softmax(logit, dim=-1)
            log_sm = repeat(log_sm, 'b l y v -> b c l y v', c=y_codebook.shape[1])
            ll_idx = repeat(y_codebook, 'b c y -> b c l y 1', l=y.shape[1])
            log_like = rearrange(torch.gather(log_sm, 4, ll_idx), 'b c l y 1 -> b c l y')
            log_like = reduce(log_like, 'b c l y -> b c l', 'sum')
            pred = log_like.argmax(1)  # [batch, seq_len]
            return pred == y  # [batch, seq_len]

    def sample_codebook(self, batch_size, device):
        """Generate a random codebook for the labels.

        Returns:
            y_codebook: [batch, tasks, y_len]
        """
        cls_int = [
            np.random.choice(
                self.config['y_vocab'] ** self.config['y_len'],
                [self.config['tasks'], 1], replace=False)
            for _ in range(batch_size)
        ]
        cls_int = torch.tensor(np.array(cls_int), device=device)
        divisor = self.config['y_vocab'] ** torch.arange(self.config['y_len'], device=device)
        return (cls_int // divisor) % self.config['y_vocab']

    @staticmethod
    def y2code(y, y_codebook):
        """Create a sequence of class codes from a tensor of class codes and a tensor of labels.

        Args:
            y (torch.Tensor): [batch, seq_len]
            y_codebook (torch.Tensor): [batch, tasks, y_len]

        Returns:
            code_seq (torch.Tensor): [batch, seq_len, y_len]
        """
        batch, seq_len = y.shape
        batch, tasks, y_len = y_codebook.shape
        code_seq = torch.gather(
            repeat(y_codebook, 'b t y -> b l t y', l=seq_len),
            2,
            repeat(y, 'b l -> b l 1 y', y=y_len)
        )
        code_seq = rearrange(code_seq, 'b l 1 y -> b l y')
        return code_seq
