import torch
import torch.nn as nn
from einops import repeat, reduce


class MamlModule(nn.Module):
    def __init__(self, reptile=False):
        super().__init__()
        self.param_inits = []
        self.params = []
        self.reptile = reptile

    def inner_update(self, loss, lr, is_meta_training=True):
        maml_modules = [module for module in self.modules() if isinstance(module, MamlModule)]
        fast_params = []
        for module in maml_modules:
            assert module.params is not None, f'Fast params are not initialized in {module}'
            for p in module.params:
                if not p.requires_grad:
                    print(f'Warning: a param in {module} does not require grad')
            fast_params.extend(module.params)

        create_graph = is_meta_training and not self.reptile
        grads = list(torch.autograd.grad(loss, fast_params, create_graph=create_graph))
        for module in maml_modules:
            new_params = []
            for p in module.params:
                g = grads.pop(0)
                new_p = p - lr * g
                if not create_graph:
                    new_p = new_p.detach()
                    new_p.requires_grad = True
                new_params.append(new_p)
            module.params = new_params
        assert len(grads) == 0, 'Not all grads are used'

    def reset_fast_params(self, batch):
        for module in self.modules():
            if isinstance(module, MamlModule):
                module.params = [repeat(param_init, '... -> b ...', b=batch) for param_init in module.param_inits]

    def reptile_update(self, lr):
        assert self.reptile, 'Reptile is not enabled'

        for module in self.modules():
            if isinstance(module, MamlModule):
                for init, param in zip(module.param_inits, module.params):
                    init.data = init.data + lr * (reduce(param.data, 'b ... -> ...', 'mean') - init.data)
