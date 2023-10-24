import torch
from torch.optim import SGD, Adam, AdamW



def create_optimizer(
    name: str,
    params: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-4
):
    if name == 'adamw':
        optimizer = AdamW(params, lr, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer