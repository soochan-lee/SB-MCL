from .cnn import CnnEncoder, CnnDecoder
from .resnet import ResNetEncoder
from .mlp import Mlp
from .maml_mlp import MamlMlp
from .maml_cnn import MamlCnnEncoder, MamlCnnDecoder
from .identity import Identity
from .unet import UNet
from .maml_unet import FastUNet

COMPONENT = {
    'CnnEncoder': CnnEncoder,
    'CnnDecoder': CnnDecoder,
    'ResNetEncoder': ResNetEncoder,
    'Mlp': Mlp,
    'MamlMlp': MamlMlp,
    'MamlCnnEncoder': MamlCnnEncoder,
    'MamlCnnDecoder': MamlCnnDecoder,
    'Identity': Identity,
    'UNet': UNet,
    'FastUNet': FastUNet,
}
