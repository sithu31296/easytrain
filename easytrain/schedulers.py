import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler
from easytrain.utils.distributed import *


def create_scheduler(
    optimizer: Optimizer,
    dataset: Dataset,
    dataloader: DataLoader,
    batch_size: int,
    epochs: int,
    lr: float,
    end_lr: float = 1e-8,
    warmup_iters: int = 10,
):
    num_of_gpus = get_world_size()
    iters_per_epoch = len(dataset) // (batch_size * num_of_gpus)
    max_iters = epochs * iters_per_epoch
    warmup_iters = min(warmup_iters, len(dataloader)-1)

    scheduler1 = LinearLR(optimizer, lr, total_iters=warmup_iters)
    scheduler2 = CosineAnnealingLR(optimizer, max_iters, end_lr)

    return ChainedScheduler([scheduler1, scheduler2])