from typing import Callable, Dict

import torch
from utils import PARAMS

METRIC_T = Callable[
    [
        torch.nn.Module,  # model
        Dict[str, torch.Tensor],  # outputs
        Dict[str, torch.Tensor],  # batch
    ],
    Dict[str, torch.Tensor],  # metric outputs
]

GET_METRIC_FN = Callable[
    [
        PARAMS,  # metric parameters
    ],
    METRIC_T,
]

# TODO add metrics

METRICS: Dict[str, GET_METRIC_FN] = {}
