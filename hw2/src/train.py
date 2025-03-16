import argparse


def train(args):
    # implement the training function here

    assert False, "Not implemented yet!"


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--data_path", type=str, help="path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=1e-5, help="learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
