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

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        # print(f"Add: {self.data} + {other.data} = {out.data}")

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
        # print(f"Sub: {self.data} - {other.data} = {out.data}")

        def grad_fn():
            # print(f"Grad: {out.grad}")
            # print(f"Self: {self.requires_grad}")
            # print(f"Other: {other.requires_grad}")
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
        # print(f"Mul: {self.data} * {other.data} = {out.data}")

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


if __name__ == "__main__":
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y + x

    z.backward()
