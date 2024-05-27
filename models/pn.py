import torch.nn as nn
from einops import rearrange, reduce

from .components import COMPONENT
from .model import Model, Output


class PN(Model):
    """Prototypical Network"""
    def __init__(self, config):
        super().__init__(config)
        x_enc_args = config['x_enc_args']
        x_enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['x_encoder']](config, x_enc_args)
        assert config['output_type'] == 'class'
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        batch, test_num = test_y.shape
        train_x_enc, test_x_enc = self.encode_x(train_x, test_x)
        prototypes = reduce(
            train_x_enc, 'b (t s) d -> b 1 t d', 'mean', t=self.config['tasks'], s=self.config['train_shots'])
        test_x_enc = rearrange(test_x_enc, 'b n d -> b n 1 d')
        logit = -reduce((test_x_enc - prototypes) ** 2, 'b n t d -> b n t', 'sum')
        loss = self.ce(rearrange(logit, 'b n t -> (b n) t'), rearrange(test_y, 'b n -> (b n)'))
        loss = reduce(loss, '(b n) -> b', 'mean', b=batch, n=test_num)

        output = Output()
        output[f'loss/meta_{meta_split}'] = loss
        if not summarize:
            return output

        output.add_classification_summary(logit, test_y, meta_split)
        return output
