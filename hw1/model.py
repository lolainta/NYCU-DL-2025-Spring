import FakeTorch as torch


class MLP:
    def __init__(self, in_features, hidden_size, out_features, activation="sigmoid"):
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, out_features)

        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise ValueError(f"Invalid activation: {activation}")

    def __call__(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)  # No activation at the output layer

    def parameters(self):
        """Automatically collect all Tensor parameters from layers"""
        params = []
        for layer in self.__dict__.values():
            if hasattr(layer, "__dict__"):  # Check if it's an object with attributes
                for param in layer.__dict__.values():
                    if isinstance(param, torch.Tensor):  # Only include Tensors
                        params.append(param)
        return params
