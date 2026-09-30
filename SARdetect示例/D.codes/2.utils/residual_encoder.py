import torch
from torch import nn
from torch.nn import functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = self.skip(value)
        value = F.relu(self.bn1(self.conv1(value)), inplace=True)
        value = self.bn2(self.conv2(value))
        return F.relu(value + identity, inplace=True)

class LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(self.stem(value))).flatten(1)
