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


def build_plot_text(learning_rate, selected_classes, number_of_epochs,
                    loss_function_name, regularization_coefficient,
                    hypersphere_radius, stochastic):
    plot_text = f'Digits: {selected_classes[0]}, {selected_classes[-1]}\n'\
                f'Rate: {learning_rate}\n'\
                f'Iterations: {number_of_epochs}\n'\
                f'Loss: {loss_function_name}\n'\
                'Variant: '
    if regularization_coefficient is not None:
        plot_text += 'RGD\n'
        plot_text += f'Coefficient: {regularization_coefficient}\n'
    elif hypersphere_radius is not None:
        plot_text += 'PGD\n'
        plot_text += f'Radius: {hypersphere_radius}\n'
    elif stochastic:
        plot_text += 'SGD\n'
    else:
        plot_text += 'GD\n'

    return plot_text


def main(learning_rate, number_of_epochs, selected_classes,
         regularization_coefficient, stochastic, hypersphere_radius,
         loss_function_name, verbose):
    if verbose:
        print('Starting...')
    # Load and preprocess data
    mnist = fetch_openml('mnist_784')

    # Choose a loss function
    loss_function = None
    if str.lower(loss_function_name) == HINGE_LOSS_STRING:
        loss_function = hinge_loss
        labels = [-1, 1]
    elif str.lower(loss_function_name) == BCE_LOSS_STRING:
        loss_function = bce_loss
        labels = [0, 1]
    else:
        raise ArgumentError("Loss function " + loss_function + " is not supported", loss_function)

    df = pd.DataFrame.from_dict(
        {'data': list(mnist['data'].astype('float64').values), 'target': mnist['target'].astype('int')})

    # filter by class, shuffle, divide to train/validation/test
    df = df.loc[df['target'].isin(selected_classes)]
    df['target'] = df['target'].replace(to_replace=selected_classes[0], value=labels[0])
    df['target'] = df['target'].replace(to_replace=selected_classes[1], value=labels[1])
    np.random.seed(1337)

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
    weights = np.random.uniform(low=0.1, high=1.0, size=(1, WEIGHT_LENGTH))

    # For plotting
    training_losses = []
    validation_losses = []

    # Training loop
    epoch_loss = 0
    previous_accuracy = 0
    for epoch in range(number_of_epochs):
        if stochastic:
            for i in range(len(train_data)):

                sample_loss, grad = loss_function(
                    np.expand_dims(train_targets[i], 0), np.expand_dims(train_data[i], 0), weights)
                epoch_loss += sample_loss
                weights = update_weights_vanilla(weights, grad, learning_rate)
            validaion_loss, _ = loss_function(validation_targets, validation_data, weights)

        else:

            epoch_loss, grads = loss_function(train_targets, train_data, weights)

            new_weights = update_weights_vanilla(weights, grads, learning_rate)
            if regularization_coefficient is not None:
                new_weights += 2 * regularization_coefficient * np.linalg.norm(new_weights)
            if hypersphere_radius is not None:
                new_weights_norm = np.linalg.norm(new_weights)
                assert new_weights_norm > 0
                new_weights *= np.sqrt(hypersphere_radius) / new_weights_norm
            weights = new_weights

        training_losses.append(epoch_loss)

        validation_epoch_loss, _ = loss_function(validation_targets, validation_data, weights)
        validation_losses.append(validation_epoch_loss)

    # if stochastic:
        # logits = sigmoid(np.dot(weights, train_data.transpose()))

    train_accuracy = binary_accuracy(train_targets, train_data, weights) * 100 # binary_accuracy(np.dot(train_data, weights), train_targets) * 100
    validation_accuracy = binary_accuracy(validation_targets, validation_data, weights)  * 100 # binary_accuracy(np.dot(validation_data, weights), validation_targets) * 100

    print(f'Train Accuracy: {train_accuracy:.2f} %, validation Accuracy: {validation_accuracy:.2f} %\n')
    print(f'Train Loss: {training_losses[-1]:.2f}, Validation Loss: {validation_losses[-1]:.2f}\n')

    # Format title
    plot_title = "Plot"

    # Format extra information
    plot_text = build_plot_text(learning_rate, selected_classes, number_of_epochs,
                                loss_function_name, regularization_coefficient,
                                hypersphere_radius, stochastic)






    plt.plot(range(number_of_epochs), training_losses)
    plt.plot(range(number_of_epochs), validation_losses)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title(plot_title)
    plt.text(0.02, 0.5, plot_text, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.3)
    plt.show()

    if verbose:
        print('Terminating...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--digits', help='The classes for binary classification (e.g., "--digits 0 9" means [0, 9])', type=int, nargs=2, default=SELECTED_CLASSES)
    parser.add_argument('-r', '--regularized', help='Use regularized gradient descent', type=float, default=None)
    parser.add_argument('-p', '--projected', help='Use projected gradient descent', type=float, default=None)
    parser.add_argument('-s', '--stochastic', help='Use stochastic gradient descent', action='store_true')
    parser.add_argument('-l', '--loss', help='Choose the loss function to use.', type=str, required=True)

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--rate', help='Learning rate', type=float, default=LEARNING_RATE)

    parser.add_argument('-v', '--verbose', help='Prints extra information and details', action='store_true')

    args = parser.parse_args()

    main(args.rate, args.epochs, args.digits,
         args.regularized, args.stochastic, args.projected,
         args.loss, args.verbose)