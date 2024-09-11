import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def sigmoid(x):
    """Compute sigmoid function. Implementation taken from https://blog.dailydoseofds.com/p/a-highly-overlooked-point-in-the"""
    return np.vectorize(scalar_sigmoid)(x)

def scalar_sigmoid(x):
    if x > 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def bce_loss(y_pred, y_true, epsilon=1e-8):
    """Compute the binary cross entropy loss averaged over the samples."""
    l = np.dot(y_pred, np.log(y_true + epsilon))+ np.dot((1-y_pred), np.log(1 - y_true + epsilon))
    return np.mean(l, axis=0)


def hinge_loss(y_pred, y_true):
    """Compute the hinge loss averaged over the samples."""
    l = np.max(0, 1 - np.dot(y_true, y_pred))
    return np.mean(l, axis=0)


def bce_grad(y_pred, y_true, x):
    """Gradient for binary cross entropy loss, expects y_pred as logit."""
    return np.dot((y_pred - y_true), x.transpose())


def update_weights_vanilla(weights, grad, learning_rate):
    return weights - learning_rate * grad
