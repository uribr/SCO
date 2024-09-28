import argparse
from argparse import ArgumentError

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
from gd_utils import *


# Default values
VALIDATION_SET_RELATIVE_SIZE = 0.2
TRAINING_SET_RELATIVE_SIZE = 0.6
TEST_SET_RELATIVE_SIZE = 0.2
REGULARIZATION_COEFFICIENT = 0.2
# +1 for the bias term
WEIGHT_LENGTH = 785

# Selecting 2 classes (digits) to make it binary
SELECTED_CLASSES = [0, 9]
LEARNING_RATE = 0.01
NUM_EPOCHS = 5

BCE_LOSS_STRING = "bce"
HINGE_LOSS_STRING = "hinge"


def main(learning_rate, number_of_epochs, selected_classes, regularization_coefficient, stochastic, hypersphere_radius, loss_function, verbose):
    if verbose:
        print('Starting...')
    # Load and preprocess data
    mnist = fetch_openml('mnist_784')

    df = pd.DataFrame.from_dict(
        {'data': list(mnist['data'].astype('float64').values), 'target': mnist['target'].astype('int')})

    # filter by class, shuffle, divide to train/validation/test
    df = df.loc[df['target'].isin(selected_classes)]
    df['target'] = df['target'].replace(to_replace=selected_classes[0], value=0)
    df['target'] = df['target'].replace(to_replace=selected_classes[1], value=1)
    np.random.seed(133652)

    df = df.sample(frac=1).reset_index(drop=True)
    # train/validation/test
    data_split = [TRAINING_SET_RELATIVE_SIZE,
                  VALIDATION_SET_RELATIVE_SIZE,
                  TEST_SET_RELATIVE_SIZE]

    num_samples = len(df)
    df_train = df.iloc[:int(num_samples * data_split[0]), :]
    df_validation = df.iloc[int(num_samples * data_split[0]): int(num_samples * (data_split[1] + data_split[0])), :]
    df_test = df.iloc[int(num_samples * (data_split[1] + data_split[0])):, :]

    train_data = np.stack(df_train['data'].to_numpy())
    validation_data = np.stack(df_validation['data'].to_numpy())

    # Adding a bias term
    train_data = np.concatenate((train_data, np.ones((len(train_data), 1), dtype='float64')), axis=1)
    validation_data = np.concatenate((validation_data, np.ones((len(validation_data), 1), dtype='float64')), axis=1)

    train_data = train_data / 255.
    validation_data = validation_data / 255.

    train_targets = df_train['target'].to_numpy()
    validation_targets = df_validation['target'].to_numpy()

    # model
    # initialize weights
    weights = np.random.uniform(low=0.01, high=1.0, size=(1, WEIGHT_LENGTH)) # np.zeros((1, WEIGHT_LENGTH)) #
    weights[-1] = 1  # Bias term

    # For plotting
    training_losses = []
    validation_losses = []

    grad_function = None
    # Choose a loss function
    if str.lower(loss_function) == HINGE_LOSS_STRING:
        loss_function = hinge_loss
    elif str.lower(loss_function) == BCE_LOSS_STRING:
        loss_function = bce_loss
    else:
        raise ArgumentError("Loss function " + loss_function + " is not supported", loss_function)

    # Training loop
    epoch_loss = 0
    previous_accuracy = 0
    for epoch in range(number_of_epochs):
        if stochastic:
            for i in range(len(train_data)):
                output = np.dot(weights, train_data[i])
                sample_logits = sigmoid(output)
                epoch_loss += bce_loss(sample_logits, train_targets[i])
                grads = bce_grad(sample_logits, train_targets[i], np.expand_dims(train_data[i], 0))
                weights = update_weights_vanilla(weights, grads, learning_rate)
            validaion_logits = sigmoid(np.dot(weights, validation_data.transpose()))

        else:
            # epoch_loss, grads = epoch_setup(weights, train_data, train_targets)
            # output = np.dot(weights, train_data.transpose())
            # logits = sigmoid(output)
            # epoch_loss = loss_function(logits, train_targets)
            epoch_loss, grads = loss_function(train_targets, train_data, weights)
            # validaion_logits = sigmoid(np.dot(weights, validation_data.transpose()))
            validaion_logits = np.dot(weights, validation_data.transpose())


            # grads = grad_function(output, train_targets, train_data)
            new_weights = update_weights_vanilla(weights, grads, learning_rate)
            if regularization_coefficient is not None:
                new_weights += 2 * regularization_coefficient * np.linalg.norm(weights)
            if hypersphere_radius is not None:
                new_weights_norm = np.linalg.norm(new_weights)
                assert new_weights_norm > 0
                new_weights *= np.sqrt(hypersphere_radius) / new_weights_norm
            weights = new_weights

        training_losses.append(epoch_loss)

        validation_epoch_loss, _= loss_function(validation_targets, validation_data, weights)
        validation_losses.append(validation_epoch_loss)

    if stochastic:
        logits = sigmoid(np.dot(weights, train_data.transpose()))

    train_accuracy = binary_accuracy(train_targets, train_data, weights) * 100 # binary_accuracy(np.dot(train_data, weights), train_targets) * 100
    validation_accuracy = binary_accuracy(validation_targets, validation_data, weights)  * 100 # binary_accuracy(np.dot(validation_data, weights), validation_targets) * 100

    print(f'Train Accuracy: {train_accuracy:.2f} %, validation Accuracy: {validation_accuracy:.2f} %\n')
    print(f'Train Loss: {training_losses[-1]:.2f}, Validation Loss: {validation_losses[-1]:.2f}\n')


    plt.plot(range(number_of_epochs), training_losses)
    plt.plot(range(number_of_epochs), validation_losses)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    if verbose:
        print('Terminating...')


# def hinge_loss2(w,x,y):
#     """ evaluates hinge loss and its gradient at w
#
#     rows of x are data points
#     y is a vector of labels
#     """
#     loss,grad = 0,0
#     for (x_,y_) in zip(x,y):
#         v = y_*np.dot(w,x_)
#         loss += max(0,1-v)
#         grad += 0 if v > 1 else -y_*x_
#     return (loss,grad)
#
# def grad_descent(x,y,w,step,thresh=0.001):
#     grad = np.inf
#     ws = np.zeros((2,0))
#     ws = np.hstack((ws,w.reshape(2,1)))
#     step_num = 1
#     delta = np.inf
#     loss0 = np.inf
#     while np.abs(delta)>thresh:
#         loss,grad = hinge_loss2(w,x,y)
#         delta = loss0-loss
#         loss0 = loss
#         grad_dir = grad/np.linalg.norm(grad)
#         w = w-step*grad_dir/step_num
#         ws = np.hstack((ws,w.reshape((2,1))))
#         step_num += 1
#     return np.sum(ws,1)/np.size(ws,1)
#
# def test1():
#     # sample data points
#     x1 = np.array((0,1,3,4,1))
#     x2 = np.array((1,2,0,1,1))
#     x  = np.vstack((x1,x2)).T
#     # sample labels
#     y = np.array((1,1,-1,-1,-1))
#     w = grad_descent(x,y,np.array((0,0)),0.1)
#     loss, grad = hinge_loss2(w,x,y)
#     plot_test(x,y,w)
#
# def plot_test(x,y,w):
#     plt.figure()
#     x1, x2 = x[:,0], x[:,1]
#     x1_min, x1_max = np.min(x1)*.7, np.max(x1)*1.3
#     x2_min, x2_max = np.min(x2)*.7, np.max(x2)*1.3
#     gridpoints = 2000
#     x1s = np.linspace(x1_min, x1_max, gridpoints)
#     x2s = np.linspace(x2_min, x2_max, gridpoints)
#     gridx1, gridx2 = np.meshgrid(x1s,x2s)
#     grid_pts = np.c_[gridx1.ravel(), gridx2.ravel()]
#     predictions = np.array([np.sign(np.dot(w,x_)) for x_ in grid_pts]).reshape((gridpoints,gridpoints))
#     plt.contourf(gridx1, gridx2, predictions, cmap=plt.cm.Paired)
#     plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
#     plt.title('total hinge loss: %g' % hinge_loss2(w,x,y)[0])
#     plt.show()

if __name__ == '__main__':
    # TODO (Uri) - Added some arguments. Will probably need to update this at some point.
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', help='The labels for binary classification (e.g., "--labels 0 9" means [0, 9])', type=int, nargs=2, default=SELECTED_CLASSES)
    parser.add_argument('-r', '--regularized', help='Use regularized gradient descent', type=float, default=None)
    parser.add_argument('-p', '--projected', help='Use projected gradient descent', type=float, default=None)
    parser.add_argument('-s', '--stochastic', help='Use stochastic gradient descent', action='store_true')
    parser.add_argument('-l', '--loss', help='Choose the loss function to use.', type=str, required=True)

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--rate', help='Learning rate', type=float, default=LEARNING_RATE)

    parser.add_argument('-v', '--verbose', help='Prints extra information and details', action='store_true')

    args = parser.parse_args()

    main(args.rate, args.epochs, args.labels,
         args.regularized, args.stochastic, args.projected,
         args.loss, args.verbose)

    # np.set_printoptions(precision=3)
    # test1()