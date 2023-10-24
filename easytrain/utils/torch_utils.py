import os
import sys
import torch
import time
import datetime
import random
import numpy as np
from torchvision import transforms as T
from collections import defaultdict, deque
from .distributed import *



def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    # total_norm = [p.grad.norm().item() for p in parameters if p.grad is not None]
    return total_norm


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    imgs, omap, metadata = tuple(zip(*batch))
    imgs = torch.stack(imgs)
    omap = torch.stack(omap)
    return imgs, omap, metadata


def denormalize(images):
    inv_normalize = T.Normalize(
        mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
        std=(1/0.229, 1/0.224, 1/0.225)
    )
    if isinstance(images, torch.Tensor):
        new_images = []
        for image in images:
            image = inv_normalize(image)
            image *= 255
            image = image.to(torch.uint8).numpy().transpose((1, 2, 0))
            new_images.append(image)
    else:
        new_images = images
    return new_images