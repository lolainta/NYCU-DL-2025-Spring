class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = 0
