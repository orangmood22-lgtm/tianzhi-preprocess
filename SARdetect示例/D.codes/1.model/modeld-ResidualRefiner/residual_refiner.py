import torch
from torch import nn
from residual_encoder import LocalEncoder
class ResidualPixelRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LocalEncoder()
        self.shared = nn.Sequential(
            nn.Linear(260, 256), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.box_delta = nn.Linear(256, 4)
        self.object_head = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor):
        features = self.encoder(image)
        features = self.shared(torch.cat([features, geometry], dim=1))
        return geometry + self.box_delta(features), self.object_head(features).squeeze(1)

