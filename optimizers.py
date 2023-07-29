import torch
from typing import Optional


OPTIMIZERS = [
    'sgd',
]

def get_optimizer(
    *,
    name: str,
    model: torch.nn.Module,
    lr: float,
    momentum: Optional[float],
    weight_decay: Optional[float],
) -> torch.optim.Optimizer:
    if name == 'sgd':
        if momentum is None or weight_decay is None:
            raise ValueError(f'Must specify parameters, got {momentum=}, {weight_decay=}')
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f'{name} optimizer not implemented')
    return optimizer