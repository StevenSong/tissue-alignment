from typing import Callable, Dict

import torch

from utils import PARAMS

GET_SCHEDULER_FN = Callable[
    [
        torch.optim.Optimizer,  # optimizer
        int,  # num epochs
        int,  # steps per epoch
        PARAMS,  # params
    ],
    torch.optim.lr_scheduler.LRScheduler,
]


def get_cosine_scheduler(
    *,  # enforce kwargs
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    scheduler_params: PARAMS,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


SCHEDULERS: Dict[str, GET_SCHEDULER_FN] = {
    "cosine": get_cosine_scheduler,
}
