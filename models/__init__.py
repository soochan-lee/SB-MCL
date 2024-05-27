from .model import Model
from .pn import PN
from .gemcl import GeMCL
from .oml import OML
from .alpaca import ALPaCA
from .oml_vae import OmlVae
from .oml_diffusion import OmlDDPM
from .std import Std
from .std_vae import StdVae
from .sbmcl import Sbmcl
from .sbmcl_vae import SbmclVae
from .std_diffusion import StdDDPM
from .sbmcl_diffusion import SbmclDDPM
from .continual_transformer import ContinualTransformer

MODEL = {
    'PN': PN,
    'GeMCL': GeMCL,
    'OML': OML,
    'Sbmcl': Sbmcl,
    'ALPaCA': ALPaCA,
    'OmlVae': OmlVae,
    'OmlDDPM': OmlDDPM,
    'SbmclVae': SbmclVae,
    'Std': Std,
    'StdVae': StdVae,
    'StdDDPM': StdDDPM,
    'SbmclDDPM': SbmclDDPM,
    'ContinualTransformer': ContinualTransformer,
}
