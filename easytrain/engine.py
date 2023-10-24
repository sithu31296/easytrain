import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from utils.logging import MetricLogger
from utils.distributed import *
from utils.torch_utils import get_total_grad_norm



def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    epoch: int,
    grad_clipping: float = 0.,
):
    model.train()

    logger = MetricLogger()

    for imgs, target in logger.log_every(dataloader, 10, f"Epoch: [{epoch}]"):
        imgs = imgs.to(device)
        target = target.to(device)
        
        pred = model(imgs)
        loss = criterion(pred, target)
        loss_value = reduce_tensor(loss).item()

        optimizer.zero_grad()
        loss.backward()

        grad_value = get_total_grad_norm(model.parameters(), norm_type=2)
        if grad_clipping > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

        optimizer.step()
        scheduler.step()

        logger.update(loss=loss_value, grad=grad_value)

    logger.synchronize_between_processes()
    print("Averaged stats: ", logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}



@torch.inference_mode()
def evaluate_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metric_fn = None,
):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_score = 0.

    logger = MetricLogger()

    for imgs, target in logger.log_every(dataloader, 30, "Test: "):
        imgs = imgs.to(device)
        target = target.to(device)

        pred = model(imgs)
        loss = criterion(pred, target)
        loss_value = reduce_dict(loss).item()

        if metric_fn is not None:
            current_score = metric_fn(pred, target)
            total_score += current_score * imgs.shape[0]
        logger.update(loss=loss_value)

    logger.synchronize_between_processes()
    print("Averaged stats: ", logger)

    stats = {k: meter.global_avg for k, meter in logger.meters.items()}

    total_score /= size
    print("Best score: ", total_score)
    return stats, total_score

