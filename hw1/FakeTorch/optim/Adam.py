import numpy as np

import FakeTorch as torch


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize state (momentum and second-moment estimates)
        self.state = {}
        for param in self.params:
            self.state[param] = {
                "step": 0,
                "m": np.zeros_like(param),  # First moment
                "v": np.zeros_like(param),  # Second moment
            }

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            state = self.state[param]
            grad = param.grad
            if self.weight_decay != 0:
                grad += self.weight_decay * param

            # Update biased first moment estimate
            state["m"] = state["m"] * self.betas[0] + grad * (1 - self.betas[0])

            # Update biased second raw moment estimate
            state["v"] = state["v"] * self.betas[1] + (grad**2) * (1 - self.betas[1])

            bias_correction1 = 1 - self.betas[0] ** (state["step"] + 1)
            bias_correction2 = 1 - self.betas[1] ** (state["step"] + 1)
            m_hat = state["m"] / (bias_correction1)
            v_hat = state["v"] / (bias_correction2)

            update = (m_hat / (v_hat**0.5 + self.eps)) * self.lr

            # Apply update
            param.data -= update

            # Increase step counter
            state["step"] += 1

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
