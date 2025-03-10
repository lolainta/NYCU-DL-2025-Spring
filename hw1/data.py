def generate_linear(n=100):
    import numpy as np

    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    import numpy as np

    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if i == 5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import os

    os.makedirs("images", exist_ok=True)

    inputs, labels = generate_linear(n=100)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap=plt.cm.RdYlBu)
    plt.savefig("images/linear.png")
    plt.clf()

    inputs, labels = generate_XOR_easy()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap=plt.cm.RdYlBu)
    plt.savefig("images/xor_easy.png")
    plt.clf()
