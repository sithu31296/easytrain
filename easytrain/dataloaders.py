import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from easytrain.utils.distributed import *
from easytrain.utils.torch_utils import fix_seed


def create_dataloader(
    trainset: Dataset,
    valset: Dataset,
    batch_size: int,
    collate_fn=None,
    num_workers: int = 8,
):
    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(trainset)
        val_sampler = DistributedSampler(valset, shuffle=False)
    else:
        train_sampler = RandomSampler(trainset)
        val_sampler = SequentialSampler(valset)
    
    train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)

    train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=collate_fn, worker_init_fn=fix_seed, pin_memory=True)
    val_loader = DataLoader(valset, 1, sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader