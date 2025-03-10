import numpy as np

from FakeTorch.Tensor import Tensor


def ReLU(x):
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

    def grad_fn():
        x.grad += ((x.data > 0).astype(np.float32) * out.grad).reshape(
            x.grad.shape
        )  # Ensure correct shape

    if out.requires_grad:
        out._grad_fn = grad_fn
        out._prev = {x}

    return out


def Sigmoid(x):
    out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)

    def grad_fn():
        x.grad += (out.data * (1 - out.data)) * out.grad  # Handles batches correctly

    if out.requires_grad:
        out._grad_fn = grad_fn
        out._prev = {x}

    return out
