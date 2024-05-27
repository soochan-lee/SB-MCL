import os
import os.path as path
import shutil
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from modulefinder import ModuleFinder

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
from train import get_config, prepare_data
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
    meta_test_set = Dataset(config, root='./data', meta_split='test')
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'],
        collate_fn=meta_test_set.collate_fn)
    train_x, train_y, test_x, test_y = next(iter(meta_test_loader))

    train_set = meta_test_set.get_tensor_dataset(train_x[0], train_y[0])
    test_set = meta_test_set.get_tensor_dataset(test_x[0], test_y[0])

    if 'online' in config and config['online']:
        # Online learning
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)  # batch_size=2 to circumvent batch norm error
        config['max_train_steps'] = len(train_loader)
    else:
        train_loader = DataLoader(
            train_set, batch_size=config['batch_size'],
            sampler=torch.utils.data.RandomSampler(
                train_set, replacement=True, num_samples=config['batch_size'] * config['max_train_steps']))
    test_loader = DataLoader(test_set, batch_size=config['eval_batch_size'], shuffle=False)

    # Main training loop
    best_step, best_test_loss = 0, torch.inf
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    for step, (train_x, train_y) in enumerate(train_loader, start=1):
        train_x, train_y = prepare_data(train_x, train_y, rank=rank)

        summarize = step % config['summary_interval'] == 0
        output = model(train_x, train_y, summarize=summarize, split='train')
        assert output['loss/train'].shape == train_x.shape[:1], 'Loss shape must be (batch_size,)'
        output['loss/train'].mean().backward()
        output['loss/train'] = output['loss/train'].detach()

        optim.step()
        optim.zero_grad()
        lr_sched.step()

        if summarize and rank == 0:
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
            train_loss = output['loss/train'].mean()
            print(f'\r[Step {step}] [{elapsed_time} / {est_total}] Train loss: {train_loss:.6f}', end='')

            if torch.isnan(train_loss).any().item():
                raise RuntimeError('NaN loss encountered')

        if step % config['eval_interval'] == 0 or step == len(train_loader):
            # Test
            print()
            model.eval()
            with torch.no_grad(), Timer('Test time: {:.3f}s'):
                output = Output()
                for test_x, test_y in test_loader:
                    test_x, test_y = prepare_data(test_x, test_y, rank=rank)
                    output = model(test_x, test_y, summarize=True, split='test')
                    output.extend(output)
                output = output.gather(world_size)

            if rank == 0:
                output.summarize(writer, step)
                test_loss = output['loss/test'].mean()
                print(f'[Step {step}] Test loss: {test_loss:.6f}')
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_step = step
            model.train()

    if rank == 0:
        # Save checkpoint
        new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step:06}.pt')
        torch.save({
            'step': step,
            'config': config,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'lr_sched': lr_sched.state_dict(),
        }, new_ckpt_path)
        print(f'\nCheckpoint saved to {new_ckpt_path}')

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
                'best_step': best_step,
                'best_test_loss': best_test_loss
            }, f)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
