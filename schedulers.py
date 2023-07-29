import torch

SCHEDULERS = [
    "cosine",
]


def get_scheduler(
    *,
    name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    else:
        raise NotImplementedError(f"{name} scheduler not implemented")
    return scheduler
