import matplotlib.pyplot as plt
import numpy as np


def show_results(x, y, pred_y, fname="result.png"):

    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict Result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.savefig(fname)
    plt.close()


def show_learning_curve(loss_curve, fname="loss.png"):

    plt.plot(loss_curve)
    plt.title("Learning Curve", fontsize=18)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.savefig(fname)
    plt.close()


def evaluate_accuracy(pred, target):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    acc = np.mean(pred == target)
    return acc
