import numpy as np

from FakeTorch.Tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
        self.b = Tensor(
            np.zeros((1, out_features)), requires_grad=True
        )  # Bias should match batch shape

    def __call__(self, x):
        return x.matmul(self.W) + self.b.data.reshape(
            1, -1
        )  # Ensures correct shape for batch
