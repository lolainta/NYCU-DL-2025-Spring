import numpy as np

import FakeTorch.Tensor as Tensor


class MSELoss:
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        loss_value = np.mean((pred.data - target.data) ** 2)
        out = Tensor(loss_value, requires_grad=True)

        def grad_fn():
            pred.grad += 2 * (pred.data - target.data) / target.data.shape[0]

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {pred}

        return out

    def __repr__(self):
        return self.__class__.__name__


class BCELoss:
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        # Apply Sigmoid to logits (numerically stable)
        sigmoid = 1 / (1 + np.exp(-pred.data))  # Sigmoid function

        # Compute BCE Loss (avoid log(0) errors)
        loss_value = -np.mean(
            target.data * np.log(sigmoid + 1e-9)
            + (1 - target.data) * np.log(1 - sigmoid + 1e-9)
        )

        out = Tensor(loss_value, requires_grad=True)

        def grad_fn():
            # Gradient of BCE Loss w.r.t. logits
            pred.grad += (sigmoid - target.data) / target.data.shape[0]

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {pred}

        return out

    def __repr__(self):
        return self.__class__.__name__
