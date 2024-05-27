import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


def cross_entropy(logits, labels):
    """Compute cross entropy loss for any shape

    Args:
        logits: [*, num_classes]
        labels: [*]

    Returns:
        [*]
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction='none'
    ).view(*labels.shape)


def angle_loss(logits, labels, min_norm=0.1):
    """Compute angle loss for any shape

    Args:
        logits: [*, 2]
        labels: (cos, sin) [*, 2]

    Returns:
        [*]
    """
    norm = torch.clamp(logits.norm(dim=-1, keepdim=True), min=min_norm)
    logits = logits / norm
    dot = torch.einsum('...d,...d->...', logits, labels)
    return 1. - dot


def vae_loss(output_mean, output_log_var, target, latent_mean, latent_log_var):
    """Compute angle loss for any shape

    Args:
        output_mean: [*, c, h, w]
        output_log_var: [*, c, ?, ?]
        target: [*, c, h, w]
        latent_mean: [*, d]
        latent_log_var: [*, d]
        include_const: whether to include the constant term in the loss

    Returns:
        [*]
    """
    output_nll = F.gaussian_nll_loss(output_mean, output_log_var, target, full=True, eps=1e-3, reduction='none')
    output_nll = reduce(output_nll, '... c h w -> ...', 'sum')

    kl = latent_mean ** 2 + torch.exp(latent_log_var) - latent_log_var
    d = kl.shape[-1]
    kl = reduce(kl, '... d -> ...', 'sum')
    kl -= d
    kl = kl * 0.5
    bpd = (output_nll + kl) / (target.shape[-1] * target.shape[-2] * target.shape[-3] * torch.log(torch.tensor(2.)))
    return bpd


OUTPUT_TYPE_TO_LOSS_FN = {
    'class': cross_entropy,
    'vector': nn.MSELoss(reduction='none'),
    'angle': angle_loss,
    'image': nn.MSELoss(reduction='none'),
}


class Timer:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self._start
        print(self.text.format(elapsed_time))


def free_dict():
    return defaultdict(free_dict)


def binarize(x):
    # Pixel values are in [-1, 1], but we want a bit darker images
    prob = x.clamp(0.0, 1.0)
    return torch.bernoulli(prob)


def kl_div(latent_mean, latent_log_var):
    kl = latent_mean ** 2 + torch.exp(latent_log_var) - latent_log_var
    kl = (reduce(kl, 'b l ... -> b l', 'sum') - latent_mean.shape[-1]) * 0.5
    return kl


def nll_to_bpd(nll, dims):
    return nll / (dims * math.log(2.))


def reparameterize(mean, log_var):
    noise = torch.randn(mean.shape, device=mean.device)
    return noise * torch.exp(log_var * 0.5) + mean
