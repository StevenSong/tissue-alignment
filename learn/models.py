from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model
from utils import PARAMS

MODEL_T = Callable[
    [
        Dict[str, torch.Tensor],  # data
    ],
    Dict[str, torch.Tensor],
]
GET_MODEL_FN = Callable[
    [
        PARAMS,  # model params
    ],
    MODEL_T,
]


class SimSiam(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone: str,
        projector_hidden_dim: int,
        predictor_hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.encoder = get_model(backbone)
        if not hasattr(self.encoder, "fc"):
            raise Exception(
                f"Unknown how to use {backbone} as backbone for SimSiam model"
            )

        self.encoder.fc = nn.Sequential(
            # projector 1
            nn.Linear(self.encoder.fc.weight.shape[1], projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            # projector 2
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            # projector 3
            nn.Linear(projector_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.predictor = nn.Sequential(
            # predictor 1
            nn.Linear(output_dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(inplace=True),
            # predictor 2
            nn.Linear(predictor_hidden_dim, output_dim),
        )

    def D(self, p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x1 = data["x1"]
        x2 = data["x2"]
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = self.D(p1, z2) / 2 + self.D(p2, z1) / 2
        return {"loss": L}


def get_simsiam(
    *,  # enforce kwargs
    model_params: PARAMS,
) -> MODEL_T:
    return SimSiam(
        backbone=model_params["backbone"],
        projector_hidden_dim=model_params["projector_hidden_dim"],
        predictor_hidden_dim=model_params["predictor_hidden_dim"],
        output_dim=model_params["output_dim"],
    )


class Triplet(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone: str,
        projector_hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.encoder = get_model(backbone)
        if not hasattr(self.encoder, "fc"):
            raise Exception(
                f"Unknown how to use {backbone} as backbone for SimSiam model"
            )

        self.encoder.fc = nn.Sequential(
            # projector 1
            nn.Linear(self.encoder.fc.weight.shape[1], projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            # projector 2
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            # projector 3
            nn.Linear(projector_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def MSE(self, x, y):
        return (x - y).pow(2).mean(dim=-1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["x"]
        pos = data["pos"]
        neg = data["neg"]

        x_embed = self.encoder(x)
        pos_embed = self.encoder(pos)
        neg_embed = self.encoder(neg)

        pos_loss = self.MSE(x_embed, pos_embed)
        neg_loss = self.MSE(x_embed, neg_embed)
        triplet_loss = (pos_loss - neg_loss + 0.001).clamp(min=0.0).mean()
        return {"loss": triplet_loss}


def get_triplet(
    *,  # enforce kwargs
    model_params: PARAMS,
) -> MODEL_T:
    return Triplet(
        backbone=model_params["backbone"],
        projector_hidden_dim=model_params["projector_hidden_dim"],
        output_dim=model_params["output_dim"],
    )


MODELS: Dict[str, GET_MODEL_FN] = {
    "simsiam": get_simsiam,
    "triplet": get_triplet,
}
