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


# def bce_loss(y, x, w, epsilon=1e-8):
#     """Compute the binary cross entropy loss averaged over the samples."""
#     loss = 0
#     grad = 0
#     for truth, sample in zip(y, x):
#         pred = np.dot(sample, w)
#         loss += -np.dot(pred, np.log(truth + epsilon)) - np.dot(1 - pred, np.log(1 - truth + epsilon))
#         grad += np.dot(pred - truth, sample)
#     return loss.item(), grad
def bce_loss(y_pred, y_true, epsilon=1e-8):
    """Compute the binary cross entropy loss averaged over the samples."""
    l = -np.dot(y_pred, np.log(y_true + epsilon)) - np.dot((1 - y_pred), np.log(1 - y_true + epsilon))
    return np.mean(l, axis=0)

def hinge_loss(y, x, w):
    """Compute the hinge loss averaged over the samples."""
    loss = 0
    grad = 0
    for (x_,y_) in zip(x,y):
        v = y_ * np.dot(w,x_)
        loss += max(0,1-v)
        grad += 0 if v > 1 else -y_*x_
    return loss.item() / len(y), grad / len(y)


def bce_grad(y_pred, y_true, x):
    """Gradient for binary cross entropy loss, expects y_pred as logit."""
    return np.dot((y_pred - y_true), x)

def hinge_grad(y_pred, y_true, x):
    """Gradient for the hinge loss, expects y_pred as logit, x as (samples x features)."""
    sample_margin = np.multiply(y_true, np.squeeze(y_pred))
    sample_grads = -y_pred.transpose() * x
    indices = np.where(sample_margin < 1)
    l = np.zeros_like(x)
    l[indices, :] = sample_grads[indices, :]
    return np.mean(l, axis=0)

def update_weights_vanilla(weights, grad, learning_rate):
    return weights - learning_rate * grad / np.linalg.norm(grad)

def binary_accuracy(y, x, w, thr=0):
    preds = np.where(np.dot(x, np.squeeze(w)) > thr, 1, -1)
    accuracy = np.sum(preds == y) / len(y)
    return accuracy