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


def bce_loss(y_true, x, w, epsilon=1e-8):
    """Compute the binary cross entropy loss averaged over the samples."""
    y_pred = np.dot(w, x.transpose())
    y_pred_logits = sigmoid(y_pred)
    l = -np.dot(y_pred_logits, np.log(y_true + epsilon)) - np.dot((1 - y_pred_logits), np.log(1 - y_true + epsilon))
    g = np.dot(y_pred_logits - y_true, x)
    return np.mean(l, axis=0), g


def hinge_loss(y, x, w):
    """Compute the hinge loss averaged over the samples."""
    loss = np.zeros(1)
    grad = np.zeros_like(w)
    for (x_,y_) in zip(x,y):
        v = y_ * np.dot(w,x_)
        loss += max(0,1-v)
        grad += 0 if v > 1 else -y_*x_
    return loss.item() / len(y), grad / len(y)


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


def binary_accuracy(y_pred, y_true, thr, labels):
    y_pred_b = np.where(np.squeeze(y_pred) > thr, labels[1], labels[0])
    accuracy = np.sum(y_pred_b == y_true) / len(y_true)
    return accuracy
