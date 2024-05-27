import torch.nn as nn


class Identity(nn.Identity):
    def __init__(self, config, argv):
        super().__init__()
        self.config = config
        if 'input_shape' in argv:
            self.input_shape = self.output_shape = argv['input_shape']
        elif 'output_shape' in argv:
            self.input_shape = self.output_shape = argv['output_shape']
        else:
            raise ValueError('Either input_shape or output_shape must be specified for Identity.')
