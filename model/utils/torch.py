import os, glob
import math
import random
from typing import Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch_geometric.graphgym.checkpoint import get_ckpt_path, get_ckpt_epoch


# Distributed Data Parallel Utilities
def all_reduce_tensor(tensor, world_size=1):
    r"""Average reduce a tensor across all workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_tensors(x, world_size=1):
    r"""Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x


# Dataloader Utilities


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


# Common Utilities


def initialize(seed=None, cudnn_deterministic=True, autograd_anomaly_detection=False):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)


def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def load_weights(model: torch.nn.Module, 
                 snapshot_dir: str):
    r"""Load weights and check keys."""
    state_dict = torch.load(snapshot_dir)
    model_dict = state_dict['model_state']
    missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

    # snapshot_keys = set(model_dict.keys())
    # model_keys = set(model.state_dict().keys())
    # missing_keys = model_keys - snapshot_keys
    # unexpected_keys = snapshot_keys - model_keys
    # if len(missing_keys) > 0:
    #     print('Missing keys:', missing_keys)
    # if len(unexpected_keys) > 0:
    #     print('Unexpected keys:', unexpected_keys)

    return missing_keys, unexpected_keys

def load_checkpoints_from_folder(model, folder, epoch=-1):
    if epoch<0:
        ckpt_folder = os.path.join(folder,'0','ckpt')
        weights_files = glob.glob(os.path.join(ckpt_folder,'*.ckpt'))
        weights_files = sorted(weights_files)
        weight_file = weights_files[-1]
    else:
        weight_file = os.path.join(folder,'0','ckpt',str(epoch)+'.ckpt')
        assert os.path.exists(weight_file), 'weight file {} not exists'.format(weight_file)
    file_name = os.path.basename(weight_file)
    epoch = int(file_name.split('.')[0])
    load_weights(model, weight_file)
    return epoch

def checkpoint_restore(model, 
                       exp_path, 
                       exp_name, 
                       epoch=0, 
                       dist=False, 
                       f='', 
                       gpu=0, 
                       optimizer: torch.optim.Optimizer = None):
    # Find file and epoch
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%04d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    # Load checkpoint and optimizer
    if len(f) > 0:
        map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
        state = torch.load(f, map_location=map_location)
        checkpoint = state if not (isinstance(state, dict) and 'state_dict' in state) else state['state_dict']

        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, 
                                                                  strict=False)
            
        if optimizer is not None:
            if isinstance(state, dict) and 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])

        print('Restore checkpoint from {} at epoch {}'.format(f, epoch))
    
    if epoch>0: epoch = epoch + 1
    return epoch, f

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num > 0 and num % multiple == 0


def is_last(num, total_num, ratio=0.95):
    return num > int(total_num * ratio)

def copy_state_dict(state_dict: OrderedDict,
                    ignore_keys: list = []):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        skip = False
        for ignore_key in ignore_keys:
            if ignore_key in k:
                skip = True
                break

        if not skip:
            new_state_dict[k] = v
    return new_state_dict

def checkpoint_save(model, 
                    optimizer, 
                    exp_path, 
                    exp_name, 
                    epoch, 
                    save_freq=16,
                    ignore_keys=['bert']):
    f = os.path.join(exp_path, exp_name + '-%04d'%epoch + '.pth')
    state_to_save = copy_state_dict(model.state_dict(), 
                                    ignore_keys=ignore_keys)
    
    state = {
        'state_dict': state_to_save, #model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, f)

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    fd = os.path.join(exp_path, exp_name + '-%04d'%epoch + '.pth')
    if os.path.isfile(fd):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(fd)

    return f


def realase_cuda(input_dict):
    for key in input_dict:
        if isinstance(input_dict[key],torch.Tensor):
            input_dict[key] = input_dict[key].detach().cpu().numpy()
    return input_dict

def fix_network_modules(model, fixed_modules=[]):
    # msg = 'fixed modules: '
    for name, params in model.named_parameters():
        if name.split('.')[0] in fixed_modules:
            params.requires_grad = False
    return model

class CosineAnnealingFunction(Callable):
    def __init__(self, max_epoch, eta_min=0.0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        next_epoch = last_epoch + 1
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * next_epoch / self.max_epoch))


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step):
        # last_step starts from -1, which means last_steps=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        else:
            if next_step > self.total_steps:
                return self.eta_min
            next_step -= self.warmup_steps
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * next_step / self.normal_steps))


def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler

def save_final_ckpt(
    model: torch.nn.Module,
    dir:str,
    optimizer= None,
    scheduler= None,
):
    r"""Saves the model checkpoint at a given epoch."""
    MODEL_STATE = 'model_state'
    OPTIMIZER_STATE = 'optimizer_state'
    SCHEDULER_STATE = 'scheduler_state'
    ckpt = {}
    ckpt[MODEL_STATE] = model.state_dict()
    if optimizer is not None:
        ckpt[OPTIMIZER_STATE] = optimizer.state_dict()
    if scheduler is not None:
        ckpt[SCHEDULER_STATE] = scheduler.state_dict()

    torch.save(ckpt,dir)

    # os.makedirs(get_ckpt_dir(), exist_ok=True)
    # torch.save(ckpt, get_ckpt_path(get_ckpt_epoch(epoch)))