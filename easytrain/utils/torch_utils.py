import os
import sys
import torch
import time
import datetime
import random
import numpy as np
from pathlib import Path
from torch import nn, Tensor
from torch.autograd import profiler
from torch.backends import cudnn
from torchvision import transforms as T
from collections import defaultdict, deque
from typing import Union
from .distributed import *



def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # total_norm = [p.grad.norm().item() for p in parameters if p.grad is not None]
    return total_norm


def fix_seed(seed: int = 123) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    imgs, omap, metadata = tuple(zip(*batch))
    imgs = torch.stack(imgs)
    omap = torch.stack(omap)
    return imgs, omap, metadata


def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = False
    cudnn.deterministic = True


def denormalize(image: torch.Tensor):
    inv_normalize = T.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    if image.ndim == 4:
        image = image.squeeze()
        
    if isinstance(image, torch.Tensor):
        image = inv_normalize(image)
        image *= 255
        image = image.to(torch.uint8).cpu().numpy().transpose((1, 2, 0))
    return image


def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6   # in MB


@torch.inference_mode()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6      # in M


@torch.no_grad()
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _  = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")