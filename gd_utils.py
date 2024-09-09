import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def bce_loss(y_pred, y_true, epsilon=1e-8):
    """Compute the binary cross entropy loss averaged over the samples."""
    l = y_pred * np.log(y_true + epsilon) + (1-y_pred) * np.log(1 - y_true + epsilon)
    return np.mean(l, axis=0)


def hinge_loss(y_pred, y_true):
    """Compute the hinge loss averaged over the samples."""
    l = np.max(0, 1 - y_true * y_pred)
    return np.mean(l, axis=0)
