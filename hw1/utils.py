def show_results(x, y, pred_y, fname="result.png"):
    import matplotlib.pyplot as plt

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
