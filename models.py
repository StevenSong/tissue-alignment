import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model


def D(p, z):
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class SimSiam(nn.Module):
    def __init__(
        self,
        *, # enforce kwargs
        backbone: str,
        projector_hidden_dim: int,
        predictor_hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.encoder = get_model(backbone)
        if not hasattr(self.encoder, 'fc'):
            raise Exception(f'Unknown how to use {backbone} as backbone for SimSiam model')

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

    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {"loss": L}
