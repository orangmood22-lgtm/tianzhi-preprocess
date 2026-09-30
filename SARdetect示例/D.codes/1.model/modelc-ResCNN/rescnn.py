import torch
from torch import nn
from residual_encoder import LocalEncoder
class ATRClassifier(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.encoder = LocalEncoder()
        self.head = nn.Linear(256, classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))

