from matplotlib import axis
from matplotlib.pylab import f
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, list):  # Convert lists to NumPy arrays
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._grad_fn = None
        self._prev = set()

    def backward(self):
        assert self.requires_grad, "This tensor has no gradient tracking!"
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        if self.grad is None:
            self.grad = np.ones_like(self.data)  # Seed gradient for scalar outputs

        for tensor in reversed(topo):
            if tensor._grad_fn:
                tensor._grad_fn()

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0)

    def item(self):
        return self.data.item()

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def grad_fn():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {self, other}
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def grad_fn():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {self, other}

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def grad_fn():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {self, other}

        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def grad_fn():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T  # Ensure correct matrix shape
            if other.requires_grad:
                other.grad += self.data.T @ out.grad  # Ensure correct matrix shape

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {self, other}

        return out

    def conv1d(self, kernel):
        other = kernel if isinstance(kernel, Tensor) else Tensor(kernel)

        # Ensure self.data and other.data are at least 2D
        self_data = self.data if self.data.ndim >= 2 else self.data.reshape(1, -1)
        other_data = other.data if other.data.ndim >= 2 else other.data.reshape(1, -1)

        batch_size, in_width = self_data.shape
        _, _, kernel_size = other_data.shape
        # print(kernel_size)

        # Create output tensor with same width as input (assuming padding="same")
        out_data = np.zeros((batch_size, in_width))

        # Perform manual 1D convolution across each batch
        for b in range(batch_size):
            for i in range(in_width):
                for k in range(kernel_size):
                    if 0 <= i - k < in_width:
                        out_data[b, i] += (
                            self_data[b, i - k] * other_data[0, k]
                        )  # Using first channel only

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

        def grad_fn():
            if self.requires_grad:
                self.grad += np.zeros_like(self.data)
                for b in range(batch_size):
                    for i in range(in_width):
                        for k in range(kernel_size):
                            if 0 <= i - k < in_width:
                                self.grad[b, i - k] += out.grad[b, i] * other_data[0, k]

            if other.requires_grad:
                other.grad += np.zeros_like(other.data)
                for k in range(kernel_size):
                    for b in range(batch_size):
                        for i in range(in_width):
                            if 0 <= i - k < in_width:
                                other.grad[0, k] += out.grad[b, i] * self_data[b, i - k]

        if out.requires_grad:
            out._grad_fn = grad_fn
            out._prev = {self, other}

        return out


if __name__ == "__main__":
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y + x

    z.backward()
