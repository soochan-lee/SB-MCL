import os
import os.path as path
import shutil
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from modulefinder import ModuleFinder

import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DATASET
from models import MODEL
from models.model import Output
from utils import Timer

parser = ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--model-config', '-mc')
parser.add_argument('--data-config', '-dc')
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-backup', action='store_true')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        includes = new_config['include'] if isinstance(new_config['include'], list) else [new_config['include']]
        for include in includes:
            include_config = get_config(include)
            config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def main():
    if torch.cuda.is_available():
        print(f'Running on {socket.gethostname()} | {torch.cuda.device_count()}x {torch.cuda.get_device_name()}')
    args = parser.parse_args()

    # Load config
    if args.config is None:
        config = get_config(args.data_config)
        model_config = get_config(args.model_config)
        config.update(model_config)
    else:
        config = get_config(args.config)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Prevent overwriting
    config['log_dir'] = args.log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {args.log_dir} already exists. Skipping...')
        exit(0)

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    if not args.no_backup:
        code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        mf = ModuleFinder([os.getcwd()])
        mf.run_script(__file__)
        for name, module in mf.modules.items():
            if module.__file__ is None:
                continue
            rel_path = path.relpath(module.__file__)
            new_path = path.join(code_dir, rel_path)
            new_dirname = path.dirname(new_path)
            os.makedirs(new_dirname, mode=0o750, exist_ok=True)
            shutil.copy2(rel_path, new_path)
        print(f'Code saved to {code_dir}')

    # Get a free port for DDP
    sock = socket.socket()
    sock.bind(('', 0))
    ddp_port = sock.getsockname()[1]
    sock.close()

    # Start DDP
    world_size = torch.cuda.device_count()
    if world_size > 1:
        assert config['batch_size'] % world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        config['batch_size'] //= world_size
        assert config['eval_batch_size'] % world_size == 0, 'Eval batch size must be divisible by the number of GPUs.'
        config['eval_batch_size'] //= world_size
        mp.spawn(train, args=(world_size, ddp_port, args, config), nprocs=world_size)
    else:
        train(0, 1, ddp_port, args, config)


def train(rank, world_size, port, args, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    if world_size > 1:
        # Initialize process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    try:
        torch.zeros([]).to(rank)  # Initialize CUDA context
    except torch.cuda.OutOfMemoryError:
        gpu_ids = os.environ['SLURM_STEP_GPUS']
        raise RuntimeError(f'GPU malfunction. Reset required for {socket.gethostname()} rank {rank} in [{gpu_ids}]')

    writer = None
    if rank == 0:
        writer = SummaryWriter(config['log_dir'], flush_secs=15)

    # Build model
    model = MODEL[config['model']](config).to(rank)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
    optim = getattr(torch.optim, config['optim'])(model.parameters(), **config['optim_args'])
    lr_sched = getattr(lr_scheduler, config['lr_sched'])(optim, **config['lr_sched_args'])
    start_step = 0

    # Resume checkpoint
    if args.resume:
        old_ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(old_ckpt_paths) > 0:
            ckpt_path = old_ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            # Get step number from checkpoint name
            start_step = int(path.basename(ckpt_path).split('-')[1].split('.')[0])
            print(f'Checkpoint loaded from {ckpt_path}')
    optim.zero_grad()

    # Data
    Dataset = DATASET[config['dataset']]
    meta_train_set = Dataset(config, root='./data', meta_split='train')
    meta_test_set = Dataset(config, root='./data', meta_split='test')
    meta_train_loader = DataLoader(
        meta_train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=meta_train_set.collate_fn)
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'],
        collate_fn=meta_test_set.collate_fn)
    meta_train_loader_iter = iter(meta_train_loader)
    meta_test_loader_iter = iter(meta_test_loader)

    # Main training loop
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    for step in range(start_step + 1, config['max_train_steps'] + 1):
        train_x, train_y, test_x, test_y = next(meta_train_loader_iter)

        batch_size = len(train_x)
        digested = 0
        outputs = []
        summarize = step % config['summary_interval'] == 0
        while batch_size - digested > 0:
            # Gradient accumulation
            bite = min(batch_size - digested, math.ceil(config['batch_size'] / config['num_bites']))
            train_x_bite = train_x[digested:digested + bite]
            train_y_bite = train_y[digested:digested + bite]
            test_x_bite = test_x[digested:digested + bite]
            test_y_bite = test_y[digested:digested + bite]
            train_x_bite, train_y_bite, test_x_bite, test_y_bite = prepare_data(
                train_x_bite, train_y_bite, test_x_bite, test_y_bite, rank=rank)

            if batch_size - digested - bite == 0:
                # Last bite
                bite_output = forward_backward(
                    model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                    batch_size=batch_size, config=config, summarize=summarize)
            else:
                with model.no_sync():
                    bite_output = forward_backward(
                        model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                        batch_size=batch_size, config=config, summarize=summarize)

            if summarize:
                outputs.append(bite_output)
            digested += bite

        optim.step()
        lr_sched.step()
        optim.zero_grad()

        if 'attn_loss' in config and config['attn_loss'] > 0 and step >= config['attn_loss_steps']:
            config['attn_loss'] = 0
            print('\nTurning off attention loss')

        if summarize:
            output = Output.cat(outputs)
            output = output.gather(world_size)

            if rank == 0:
                writer.add_scalar('lr', lr_sched.get_last_lr()[0], step)
                output.summarize(writer, step)

                # Compute remaining time
                now = datetime.now()
                elapsed_time = now - start_time
                elapsed_steps = step - start_step
                total_steps = config['max_train_steps'] - start_step
                est_total = elapsed_time * total_steps / elapsed_steps
                # Remove microseconds for brevity
                elapsed_time = str(elapsed_time).split('.')[0]
                est_total = str(est_total).split('.')[0]
                meta_train_loss = output['loss/meta_train'].mean()
                print(f'\r[Step {step}] [{elapsed_time} / {est_total}] Meta-train loss: {meta_train_loss:.6f}', end='')

                if torch.isnan(meta_train_loss).any().item():
                    raise RuntimeError('NaN loss encountered')

        if rank == 0 and step % config['ckpt_interval'] == 0:
            old_ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))

            new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step:06}.pt')
            torch.save({
                'step': step,
                'config': config,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_sched': lr_sched.state_dict(),
            }, new_ckpt_path)
            print(f'\nCheckpoint saved to {new_ckpt_path}')

            # Remove old checkpoints
            for ckpt_path in old_ckpt_paths:
                os.remove(ckpt_path)

        if step % config['eval_interval'] == 0:
            # Meta-test
            print()
            model.eval()
            with torch.no_grad(), Timer('Meta-test time: {:.3f}s'):
                output = Output()
                for _ in range(config['eval_iters']):
                    train_x, train_y, test_x, test_y = next(meta_test_loader_iter)

                    batch_size = len(train_x)
                    digested = 0
                    while batch_size - digested > 0:
                        bite = min(batch_size - digested, math.ceil(config['eval_batch_size'] / config['num_bites']))
                        train_x_bite = train_x[digested:digested + bite]
                        train_y_bite = train_y[digested:digested + bite]
                        test_x_bite = test_x[digested:digested + bite]
                        test_y_bite = test_y[digested:digested + bite]
                        train_x_bite, train_y_bite, test_x_bite, test_y_bite = prepare_data(
                            train_x_bite, train_y_bite, test_x_bite, test_y_bite, rank=rank)

                        bite_output = model(
                            train_x_bite, train_y_bite, test_x_bite, test_y_bite, summarize=True, meta_split='test')
                        output.extend(bite_output)
                        digested += bite

                if summarize:
                    output = output.gather(world_size)

                    if rank == 0:
                        output.summarize(writer, step)
                        meta_test_loss = output['loss/meta_test'].mean()
                        print(f'[Step {step}] Meta-test loss: {meta_test_loss:.6f}')

            model.train()

    if rank == 0:
        writer.flush()
        end_time = datetime.now()
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
        with open(path.join(config['log_dir'], 'completed.yaml'), 'a') as f:
            yaml.dump({
                'step': step,
                'end_time': end_time,
            }, f)

    if world_size > 1:
        dist.destroy_process_group()


def forward_backward(model, train_x, train_y, test_x, test_y, batch_size, config, summarize=False):
    output = model(train_x, train_y, test_x, test_y, summarize=summarize, meta_split='train')
    loss_sum = output['loss/meta_train'].sum()
    if 'attn_loss_sum/meta_train' in output and config['attn_loss'] > 0:
        loss_sum = loss_sum + config['attn_loss'] * output['attn_loss_sum/meta_train'].sum()
    scaled_loss = loss_sum / batch_size
    scaled_loss.backward()
    output['loss/meta_train'] = output['loss/meta_train'].detach()
    return output


def prepare_data(*data, rank=0):
    prepared_data = []
    for d in data:
        if isinstance(d, torch.Tensor):
            d = d.to(rank)

            if d.dtype == torch.uint8:
                d = d.float() * 2 / 255 - 1
        prepared_data.append(d)
    return prepared_data


if __name__ == '__main__':
    main()
