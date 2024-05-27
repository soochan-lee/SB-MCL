from einops import reduce, rearrange
from torch import nn

from models.components import COMPONENT, Mlp
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN


class Std(nn.Module):
    """Standard model"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['encoder']](config, enc_args)

        dec_args = config['dec_args']
        dec_args['output_shape'] = [config['tasks']] if config['output_type'] == 'class' else config['y_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        mlp_args = config['mlp_args']
        mlp_args['input_shape'] = self.encoder.output_shape
        mlp_args['output_shape'] = self.decoder.input_shape
        self.mlp = Mlp(config, mlp_args)

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, x, y, summarize, split):
        x = rearrange(x, 'b ... -> b 1 ...')
        y = rearrange(y, 'b ... -> b 1 ...')
        logit = self.decoder(self.mlp(self.encoder(x)))
        loss = reduce(self.loss_fn(logit, y), 'b ... -> b', 'mean')

        output = Output()
        output[f'loss/{split}'] = loss
        if not summarize:
            return output

        if self.config['output_type'] == 'class':
            output.add_classification_summary(logit, y, split)
        elif self.config['output_type'] == 'image':
            output.add_image_comparison_summary(
                rearrange(x, 'b 1 ... -> 1 b ...'),
                rearrange(y, 'b 1 ... -> 1 b ...'),
                rearrange(x, 'b 1 ... -> 1 b ...'),
                rearrange(logit, 'b 1 ... -> 1 b ...'),
                key=f'completion/{split}')
        return output
