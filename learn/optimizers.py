from typing import Callable, Dict, Union

import torch
from utils import PARAMS

GET_OPTIMIZER_FN = Callable[
    [
        torch.nn.Module,  # model
        float,  # learning rate
        PARAMS,  # params
    ],
    torch.optim.Optimizer,
]


def get_sgd(
    *,  # enforce kwargs
    model: torch.nn.Module,
    lr: float,
    optimizer_params: PARAMS,
) -> torch.optim.Optimizer:
    for param in ["momentum", "weight_decay"]:
        if param not in optimizer_params:
            raise ValueError(f"Must specify {param} optimizer param")
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=optimizer_params["momentum"],
        weight_decay=optimizer_params["weight_decay"],
    )


def get_adam(
    *,  # enforce kwargs
    model: torch.nn.Module,
    lr: float,
    optimizer_params: PARAMS,
) -> torch.optim.Optimizer:
    # TODO: implement adam params
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )


OPTIMIZERS: Dict[str, GET_OPTIMIZER_FN] = {
    "sgd": get_sgd,
    "adam": get_adam,
}
