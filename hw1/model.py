import FakeTorch as torch


class MLP:
    def __init__(self, in_features, hidden_size, out_features):
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, out_features)

    def __call__(self, x):
        x = torch.nn.Sigmoid(self.fc1(x))
        x = torch.nn.Sigmoid(self.fc2(x))
        return self.fc3(x)  # No activation at the output layer

    def parameters(self):
        return [self.fc1.W, self.fc1.b, self.fc2.W, self.fc2.b, self.fc3.W, self.fc3.b]
