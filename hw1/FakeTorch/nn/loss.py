import numpy as np

import FakeTorch.Tensor as Tensor


def MSELoss(pred, target):
    diff = pred - target
    out = Tensor((diff.data**2), requires_grad=True)

    def grad_fn():
        # pred.grad += (2 * diff.data / target.data.size) * out.grad
        # pred.grad += (2 * diff.data / target.data.shape[0]) * out.grad
        pred.grad += (2 * diff.data / target.data.shape[0]) * out.grad.reshape(-1, 1)

    if out.requires_grad:
        out._grad_fn = grad_fn
        out._prev = {pred}
        out.grad = 1.0

    return out


def BCELoss(pred, target):
    """
    Binary Cross Entropy (BCE) Loss for binary classification.

    pred: Tensor, raw logits (N, 1)
    target: Tensor, binary labels (N, 1) (0 or 1)
    """
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
