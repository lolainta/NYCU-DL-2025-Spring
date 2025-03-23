import torch
from torch import nn


class DyT2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        x = x * self.alpha.view(1, C, 1)
        x = torch.tanh(x)
        x = x * self.gamma.view(1, C, 1)
        x = x + self.beta.view(1, C, 1)
        x = x.view(B, C, H, W)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            # DyT2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding="same", bias=False
            ),
            nn.BatchNorm2d(out_channels),
            # DyT2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
