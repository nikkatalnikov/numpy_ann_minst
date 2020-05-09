import numpy as np


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_dx(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_dx(x):
    return np.where(x > 0, 1.0, 0.0)


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)
