import numpy as np


import FakeTorch as torch

from model import MLP
from utils import show_results
from data import generate_linear, generate_XOR_easy


np.random.seed(0)
np.set_printoptions(precision=3, linewidth=400, suppress=True)


def main():
    # Dummy dataset: 10 samples, 3 input features
    x_org, y_org = generate_XOR_easy()
    # x_org, y_org = generate_linear()

    X = torch.tensor(x_org)
    Y = torch.tensor(y_org)

    print(f"X: {X.data.shape}")
    print(f"Y: {Y.data.shape}")

    # Create model
    model = MLP(2, 64, 1)

    # Training parameters
    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):
        for i in range(X.data.shape[0]):
            x = torch.tensor(X.data[i].reshape(1, -1))
            y = torch.tensor(Y.data[i].reshape(1, -1))

            # Forward pass
            pred = model(x)
            loss = torch.nn.BCELoss()(pred, y)
            loss = torch.nn.MSELoss()(pred, y)

            # Backward pass
            for p in model.parameters():
                p.zero_grad()
            loss.backward()

            # Update weights
            for p in model.parameters():
                p.data -= learning_rate * p.grad

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.data.item()}")
            show_results(
                x_org, y_org, model(X).data, fname=f"results/linear_torh_{epoch}.png"
            )

    show_results(x_org, y_org, model(X).data, fname="results/linear_torh_final.png")


if __name__ == "__main__":
    main()
