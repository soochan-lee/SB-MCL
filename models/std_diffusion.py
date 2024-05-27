import torch
from einops import reduce, repeat, rearrange

from models.components import COMPONENT
from models.components.diffusion import Diffusion
from models.model import Output


class StdDDPM(Diffusion):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.set_denoiser(config)

    def set_denoiser(self, config):
        # UNet
        unet_args = config['unet_args']
        unet_args['img_channels'] = config['x_shape'][0]
        self.denoiser = COMPONENT[config['backbone']](**unet_args)

    def forward(self, x, y, summarize, split):
        if split == 'test':
            x = repeat(x, 'b ... -> b l ...', l=self.config['eval_t_batch'])
        else:
            x = rearrange(x, 'b ... -> b 1 ...')
        batch, inner_batch = x.shape[:2]

        it = 1 if split == 'train' else self.config['eval_t_num'] // self.config['eval_t_batch']
        loss_sum = 0
        for i in range(it):
            # Forward process
            t = torch.randint(low=0, high=self.n_times, size=(batch, inner_batch), device=x.device)
            noisy_x, noise = self.make_noisy(x, t)
            pred_noise = self.denoiser(noisy_x, t)

            loss = self.loss_fn(pred_noise, noise)
            loss = reduce(loss, 'b l c h w -> b', 'mean')
            loss_sum = loss_sum + loss
        loss = loss_sum / it

        output = Output()
        output[f'loss/{split}'] = loss
        if not summarize:
            return output

        if split == 'train':
            # Evaluation
            generated_images = self.generate(8)
            output.add_image_comparison_summary(generated_images, key=f'generation/{split}')

        return output
