import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler, MultiStepLR
from easytrain.utils.distributed import *


def create_scheduler(
    scheduler_name: str,
    optimizer: Optimizer,
    dataloader: DataLoader,
    batch_size: int,
    epochs: int,
    lr: float,
    end_lr: float = 1e-8,
    warmup_iters: int = 10,
    milestones: list = None,
):
    num_of_gpus = get_world_size()
    iters_per_epoch = len(dataloader.dataset) // (batch_size * num_of_gpus)
    max_iters = epochs * iters_per_epoch
    warmup_iters = min(warmup_iters, len(dataloader)-1)

    scheduler1 = LinearLR(optimizer, lr, total_iters=warmup_iters)

    if scheduler_name == 'multisteplr':
        if milestones is None:
            milestones = [int(epochs*0.6*iters_per_epoch), int(epochs*0.85*iters_per_epoch)]
        else:
            milestones = list(map(lambda x: int(x*iters_per_epoch), milestones))
        scheduler2 = MultiStepLR(optimizer, milestones, gamma=0.1)
    else:
        scheduler2 = CosineAnnealingLR(optimizer, max_iters, end_lr)

    return ChainedScheduler([scheduler1, scheduler2])