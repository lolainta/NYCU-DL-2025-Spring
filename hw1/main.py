import os
import numpy as np
from multiprocessing import Process


import FakeTorch as torch

from model import MLP
from utils import show_results, show_learning_curve, evaluate_accuracy
from data import generate_linear, generate_XOR_easy


np.set_printoptions(precision=3, linewidth=400, suppress=True)


def main(
    dataset: str,
    hidden_size,
    learning_rate,
    epochs,
    loss_fn,
    activation,
    optim,
    verbose=True,
):
    # Set random seed
    np.random.seed(0)

    # Print parameters
    if verbose:
        print(f"Params: {locals()}")
    out_dir = f"hidden_{hidden_size}_lr_{learning_rate}_epochs_{epochs}_{loss_fn}_{activation}_{optim}_{dataset}"
    if verbose:
        print(f"Output directory: {out_dir}")
    out_dir = os.path.join("results", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Generate data
    if dataset == "linear":
        data = generate_linear()
    elif dataset == "xor":
        data = generate_XOR_easy()
    else:
        raise ValueError(f"Invalid data: {data}")
    x_org, y_org = data
    X = torch.tensor(x_org)
    Y = torch.tensor(y_org)

    # Create model
    model = MLP(2, hidden_size, 1, activation)

    # Create optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optim}")

    loss_curve = []

    for epoch in range(epochs):
        for i in range(X.data.shape[0]):
            x = torch.tensor(X.data[i].reshape(1, -1))
            y = torch.tensor(Y.data[i].reshape(1, -1))

            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

        loss = loss_fn(model(X), Y)
        loss_curve.append(loss.data.item())

        if (epoch + 1) % 100 == 0:
            if verbose:
                print(f"{out_dir}: Epoch {epoch + 1}, Loss: {loss.data.item()}")

            show_results(
                x_org, y_org, model(X).data, fname=f"{out_dir}/epoch-{epoch+1}.png"
            )

    show_results(x_org, y_org, model(X).data, fname=f"{out_dir}/final.png")
    show_learning_curve(loss_curve, fname=f"{out_dir}/loss.png")

    # Evaluate model
    Y_pred = model(X).data
    acc = evaluate_accuracy(Y_pred, Y.data)
    if verbose:
        print(f"Accuracy: {acc}")

    with open(f"{out_dir}/accuracy.txt", "w") as f:
        f.write(f"{acc*100:.2f}\\%")
    print(f"{out_dir}: Accuracy: {acc}")


if __name__ == "__main__":
    plist = []
    for hidden_size in [1, 2, 4, 8, 16, 32]:
        for learning_rate in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            for epochs in [500, 1000, 2000]:
                if epochs != 2000:
                    continue
                for loss_fn in [torch.nn.BCELoss(), torch.nn.MSELoss()]:
                    for dataset in reversed(["linear", "xor"]):
                        for activation in ["sigmoid", "relu", "tanh", "none"]:
                            for optim in ["SGD", "Adam"]:
                                p = Process(
                                    target=main,
                                    args=(
                                        dataset,
                                        hidden_size,
                                        learning_rate,
                                        epochs,
                                        loss_fn,
                                        activation,
                                        optim,
                                        False,
                                    ),
                                )
                                p.start()
                                plist.append(p)

    for p in plist:
        p.join()
