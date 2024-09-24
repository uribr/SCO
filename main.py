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
    np.random.seed(42)

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
    weight_length = WEIGHT_LENGTH
    weights = np.random.uniform(size=(1, weight_length))
    weights[-1] = 1  # Bias term

    # For plotting
    training_losses = []
    validation_losses = []

    grad_function = None
    # Choose a loss function
    if loss_function is HINGE_LOSS_STRING:
        loss_function = hinge_loss
        grad_function = hinge_grad
    elif loss_function is BCE_LOSS_STRING:
        loss_function = bce_loss
        grad_function = bce_grad
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
            output = np.dot(weights, train_data.transpose())
            logits = sigmoid(output)
            epoch_loss = loss_function(logits, train_targets)
            validaion_logits = sigmoid(np.dot(weights, validation_data.transpose()))

            grads = grad_function(logits, train_targets, train_data)
            new_weights = update_weights_vanilla(weights, grads, learning_rate)
            if regularization_coefficient is not None:
                new_weights += 2 * regularization_coefficient * np.linalg.norm(weights)
            if hypersphere_radius is not None:
                new_weights_norm = np.linalg.norm(new_weights)
                assert new_weights_norm > 0
                new_weights *= np.sqrt(hypersphere_radius) / new_weights_norm
            weights = new_weights

        epoch_loss = epoch_loss / len(train_data)
        training_losses.append(epoch_loss)

        validation_epoch_loss = (bce_loss(validaion_logits, validation_targets)
                                 / len(validation_data))
        validation_losses.append(validation_epoch_loss)

    if stochastic:
        logits = sigmoid(np.dot(weights, train_data.transpose()))

    train_accuracy = binary_accuracy(logits, train_targets) * 100
    validation_accuracy = binary_accuracy(validaion_logits, validation_targets) * 100

    print(f'Train Accuracy: {train_accuracy:.2f} %, validation Accuracy: {validation_accuracy:.2f} %\n')
    print(f'Train Loss: {epoch_loss:.2f}, Validation Loss: {validation_epoch_loss:.2f}\n')


    plt.plot(range(NUM_EPOCHS), training_losses)
    plt.plot(range(NUM_EPOCHS), validation_losses)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    if verbose:
        print('Terminating...')


if __name__ == '__main__':
    # TODO (Uri) - Added some arguments. Will probably need to update this at some point.
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', help='The labels for binary classification (e.g., "--labels 0 9" means [0, 9])', type=int, nargs=2, default=SELECTED_CLASSES)
    parser.add_argument('-r', '--regularized', help='Use regularized gradient descent', type=float, default=None)
    parser.add_argument('-p', '--projected', help='Use projected gradient descent', type=float, default=None)
    parser.add_argument('-s', '--stochastic', help='Use stochastic gradient descent', action='store_true')
    parser.add_argument('-l', '--loss', help='Choose the loss function to use.', type=str,required=true)

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--rate', help='Learning rate', type=float, default=LEARNING_RATE)

    parser.add_argument('-v', '--verbose', help='Prints extra information and details', action='store_true')

    args = parser.parse_args()

    main(args.rate, args.epochs, args.labels,
         args.regularized, args.stochastic, args.projected,
         args.loss, args.verbose)