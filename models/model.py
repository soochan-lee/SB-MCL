import torch
import torch.nn as nn
from einops import rearrange, reduce, pack, unpack
from torch import distributed as dist


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def encode_x(self, train_x, test_x):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]

        x = torch.cat([train_x, test_x], dim=1)
        x_enc = self.x_encoder(x)
        train_x_enc, test_x_enc = torch.split(x_enc, [train_num, test_num], dim=1)
        return train_x_enc, test_x_enc


class Output(dict):
    @staticmethod
    def cat(outputs):
        output = Output()
        for k in outputs[0].keys():
            output[k] = torch.cat([o[k] for o in outputs], dim=0)
        return output

    def extend(self, other):
        for k, v in other.items():
            if k not in self:
                self[k] = v
            elif len(v.shape) == 4:
                # Keep only one image to save memory
                self[k] = v
            else:
                self[k] = torch.cat([self[k], v], dim=0)

    def gather(self, world_size):
        """Gather tensors from DDP processes"""
        if world_size == 1:
            return self

        gathered = Output()
        for k, v in self.items():
            gathered_v = torch.zeros((world_size,) + v.shape, dtype=v.dtype, device=v.device)
            dist.all_gather_into_tensor(gathered_v, v.detach())
            gathered[k] = rearrange(gathered_v, 'w b ... -> (w b) ...')
        return gathered

    def summarize(self, writer, step):
        for key, value in self.items():
            if len(value.shape) == 1:
                writer.add_scalar(key, value.mean().item(), step)
            elif len(value.shape) == 4:
                writer.add_image(key, value[0], step)
            elif len(value.shape) == 5:
                writer.add_image(key, value[0][0], step)
            else:
                raise NotImplementedError(f'Cannot summarize {key} with shape {value.shape}')

    def export(self):
        result = {}
        for key, value in self.items():
            if len(value.shape) == 1:
                result[key] = value.mean().item()
            elif len(value.shape) == 4:
                result[key] = value[0].cpu().numpy()
            elif len(value.shape) == 5:
                result[key] = value[0][0].cpu().numpy()
            else:
                raise NotImplementedError
        return result

    def add_classification_summary(self, logit, test_y, meta_split):
        self[f'acc/{meta_split}'] = reduce((logit.argmax(-1) == test_y).float(), 'b l -> b', 'mean')

    def add_image_comparison_summary(self, *images, key, num_samples=None):
        _, num, c, h, w = images[0].shape
        if num_samples is None:
            num_samples = min(8, num)
        indices = torch.randperm(num)[:num_samples]
        comparison, _ = pack([image.detach()[0, indices] for image in images], 'n c * w')
        comparison = rearrange(comparison, 'n c h w -> 1 c h (n w)')
        self[key] = comparison

    def add_sbmcl_prior_summary(self, prior_mean, prior_log_pre):
        prior_mean = prior_mean.detach()
        prior_log_pre = prior_log_pre.detach()
        self['prior/mean/mean'] = rearrange(prior_mean.mean(), '-> 1')
        self['prior/mean/std'] = rearrange(prior_mean.std(), '-> 1')
        self['prior/mean/min'] = rearrange(prior_mean.min(), '-> 1')
        self['prior/mean/max'] = rearrange(prior_mean.max(), '-> 1')
        self['prior/log_pre/mean'] = rearrange(prior_log_pre.mean(), '-> 1')
        self['prior/log_pre/std'] = rearrange(prior_log_pre.std(), '-> 1')
        self['prior/log_pre/min'] = rearrange(prior_log_pre.min(), '-> 1')
        self['prior/log_pre/max'] = rearrange(prior_log_pre.max(), '-> 1')
