import numpy as np

from FakeTorch.Tensor import Tensor


class ReLU:
    def __call__(self, x):
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

        def grad_fn():
            if x.grad is not None:
                x.grad += ((x.data > 0).astype(np.float32) * out.grad).reshape(
                    x.grad.shape
                )

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {x}

        return out


class Sigmoid:
    def __call__(self, x):
        out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)

        def grad_fn():
            if x.grad is not None:
                x.grad += (
                    out.data * (1 - out.data)
                ) * out.grad  # Ensure proper broadcasting

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {x}

        return out
