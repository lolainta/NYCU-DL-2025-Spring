import numpy as np

import FakeTorch as torch


class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.W = torch.tensor(
            np.random.randn(out_channels, in_channels, kernel_size), requires_grad=True
        )
        self.b = torch.tensor(np.zeros((out_channels, 1)), requires_grad=True)
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        # print(f"Conv1d: {x.shape} -> {self.W.shape}")
        return x.conv1d(self.W) + self.b.data
