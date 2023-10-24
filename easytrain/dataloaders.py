import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler, BatchSampler
from utils.distributed import *
from utils.torch_utils import fix_seed


def create_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn = None,
):
    if is_dist_avail_and_initialized():
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    train_batch_sampler = BatchSampler(train_sampler, batch_size, True)

    return DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True, collate_fn=collate_fn, worker_init_fn=fix_seed), train_sampler


def create_val_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn = None,
):
    test_sampler = SequentialSampler(dataset)
    return DataLoader(dataset, batch_size, sampler=test_sampler, num_workers=4, drop_last=False, pin_memory=True, collate_fn=collate_fn)