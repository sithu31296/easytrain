import torch
from torch.optim import SGD, Adam, AdamW



def create_optimizer(
    name: str,
    model: torch.nn.Module,
    lr: float,
    params = None,
    weight_decay: float = 1e-4
):
    if params is not None:
        params = [p for p in model.parameters() if p.requires_grad]
    if name == 'adamw':
        optimizer = AdamW(params, lr, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(params, lr, weight_decay=weight_decay, momentum=0.9)
    return optimizer