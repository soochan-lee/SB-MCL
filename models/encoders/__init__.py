from .cnn import CnnEncoder
from .resnet import ResNetEncoder
from .mlp import MlpEncoder

X_ENCODER = {
    'CnnEncoder': CnnEncoder,
    'ResNetEncoder': ResNetEncoder,
    'MlpEncoder': MlpEncoder,
}
