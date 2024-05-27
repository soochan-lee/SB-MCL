import math
import torch
from einops import reduce, rearrange
from torch import nn

from models import Model
from models.components import COMPONENT
from models.components.maml_mlp import MamlMlp
from models.maml_nn import MamlModule, MamlLinear
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN


class OML(Model, MamlModule):
    def __init__(self, config):
        Model.__init__(self, config)
        MamlModule.__init__(self, reptile=config['reptile'])
        self.log_inner_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.x_encoder = COMPONENT[config['encoder']](config, enc_args)

        dec_args = config['dec_args']
        dec_args['output_shape'] = [config['tasks']] if config['output_type'] == 'class' else config['y_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        maml_mlp_args = config['maml_mlp_args']
        maml_mlp_args['input_shape'] = self.x_encoder.output_shape
        maml_mlp_args['output_shape'] = self.decoder.input_shape
        maml_mlp_args['reptile'] = config['reptile']
        if config['output_type'] == 'class':
            maml_mlp_args['output_shape'] = [1000]
        self.maml_mlp = MamlMlp(config, maml_mlp_args)

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        batch, train_num = train_x.shape[:2]
        batch, test_num = test_x.shape[:2]
        is_meta_training = meta_split == 'train'

        if self.config['output_type'] == 'class':
            # Reset the last linear layer
            last_maml_linear = self.maml_mlp.mlp[-1]
            assert isinstance(last_maml_linear, MamlLinear)
            last_maml_linear.initialize()

        inner_lr = self.log_inner_lr.exp()
        with torch.enable_grad():
            self.reset_fast_params(batch)

        #########
        # Train #
        #########
        start = 0
        chunk = self.config['train_chunk'] if 'train_chunk' in self.config else train_num
        while start < train_num:
            end = min(start + chunk, train_num)
            train_x_chunk = train_x[:, start:end]
            train_y_chunk = train_y[:, start:end]
            self.forward_train(train_x_chunk, train_y_chunk, inner_lr, is_meta_training)
            start = end

        ########
        # Test #
        ########
        test_x_enc = self.x_encoder(test_x)
        mlp_out = self.maml_mlp(test_x_enc)
        logit = self.decoder(mlp_out)
        meta_loss = reduce(self.loss_fn(logit, test_y), 'b ... -> b', 'mean')

        if self.reptile and is_meta_training:
            self.reptile_update(self.config['reptile_lr'])

        output = Output()
        output[f'loss/meta_{meta_split}'] = meta_loss
        if not summarize:
            return output

        if meta_split == 'train':
            output['lr_inner'] = rearrange(inner_lr.detach(), '-> 1')

        if self.config['output_type'] == 'class':
            output.add_classification_summary(logit, test_y, meta_split)
        elif self.config['output_type'] == 'image':
            output.add_image_comparison_summary(test_x, test_y, test_x, logit, key=f'completion/meta_{meta_split}')
        return output

    def forward_train(self, train_x, train_y, inner_lr, is_meta_training):
        batch, train_num = train_x.shape[:2]
        train_x_enc = self.x_encoder(train_x)

        with torch.enable_grad():
            for i in range(train_num):
                # Sequentially forward training data
                x_i = train_x_enc[:, i:i + 1]
                y_i = train_y[:, i:i + 1]
                mlp_out = self.maml_mlp(x_i)
                logit = self.decoder(mlp_out)
                loss = reduce(self.loss_fn(logit, y_i), 'b ... -> b', 'mean').sum()
                self.inner_update(loss, inner_lr, is_meta_training=is_meta_training)

