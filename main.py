import time
import argparse
from argparse import ArgumentError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import plot_utils
from gd_utils import *
from plot_utils import *
from main_utils import *
from configuration import *

def do_stuff(learning_rate, number_of_epochs, selected_classes,
         regularization_coefficient, stochastic, hypersphere_radius,
         loss_function_name, verbose):

    # TODO - Move the code of main to an auxiliary function and instead iterate over the configurations and run them
    if verbose:
        print('Starting...')
    # Load and preprocess data
    mnist = fetch_openml('mnist_784')

    # Choose a loss function
    loss_function = None
    if str.lower(loss_function_name) == HINGE_LOSS_STRING:
        loss_function = hinge_loss
        labels = [-1, 1]
        pred_thr = 0
    elif str.lower(loss_function_name) == BCE_LOSS_STRING:
        loss_function = bce_loss
        labels = [0, 1]
        pred_thr = 0.5
    else:
        raise ArgumentError(f"Loss function {loss_function_name} is not supported")

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
    weights = np.random.randn(WEIGHT_LENGTH) * np.sqrt(1 / WEIGHT_LENGTH)
    weights[-1] = 0

    # For plotting
    training_losses = []
    validation_losses = []

    # Training loop
    epoch_loss = 0
    previous_accuracy = 0
    use_projection = hypersphere_radius is not None
    use_regularization = regularization_coefficient is not None
    for epoch in range(number_of_epochs):
        if stochastic:
            for i in range(len(train_data)):

                sample_loss, grad = loss_function(
                    np.expand_dims(train_targets[i], 0), np.expand_dims(train_data[i], 0), weights)
                epoch_loss += sample_loss
                weights = update_weights_vanilla(weights, grad, learning_rate)
            epoch_loss = epoch_loss / len(train_data)
            validaion_loss, _ = loss_function(validation_targets, validation_data, weights)

        else:

            epoch_loss, grads = loss_function(train_targets, train_data, weights)

            new_weights = update_weights_vanilla(weights, grads, learning_rate)

            if use_regularization:
                new_weights += 2 * regularization_coefficient * np.linalg.norm(weights)
            if use_projection:
                new_weights = np.clip(new_weights, -hypersphere_radius, hypersphere_radius)

            weights = new_weights

            if verbose:
                if (use_regularization or use_projection) and epoch % REPORT_FREQUENCY == 0:
                    print(f'Weights Norm: {np.linalg.norm(weights)}')

        training_losses.append(epoch_loss)

        validation_epoch_loss, _ = loss_function(validation_targets, validation_data, weights)
        validation_losses.append(validation_epoch_loss)

    # if stochastic:
        # logits = sigmoid(np.dot(weights, train_data.transpose()))

    train_y_pred = np.dot(train_data, weights.transpose())
    validation_y_pred = np.dot(validation_data, weights.transpose())

    if loss_function_name == BCE_LOSS_STRING:
        train_y_pred = sigmoid(train_y_pred)
        validation_y_pred = sigmoid(validation_y_pred)

    train_accuracy = binary_accuracy(train_y_pred, train_targets, pred_thr, labels) * 100
    validation_accuracy = binary_accuracy(validation_y_pred, validation_targets, pred_thr, labels) * 100
    # train_accuracy = binary_accuracy(train_targets, train_data, weights) * 100
    # validation_accuracy = binary_accuracy(validation_targets, validation_data, weights)  * 100

    print(f'Train Accuracy: {train_accuracy:.3f} %, validation Accuracy: {validation_accuracy:.3f} %\n')
    print(f'Train Loss: {training_losses[-1]:.3f}, Validation Loss: {validation_losses[-1]:.3f}\n')

    # Format title
    plot_title = "Loss vs. Iterations"

    # Format extra information
    plot_text = build_plot_text(learning_rate, selected_classes, number_of_epochs,
                                loss_function_name, regularization_coefficient,
                                hypersphere_radius, stochastic, training_losses[-1],
                                validation_losses[-1], train_accuracy, validation_accuracy)

    # TODO - Add the test set to the plots or redistribute the data into just train and test (replacing validation with test)
    plt.plot(range(number_of_epochs), training_losses)
    plt.plot(range(number_of_epochs), validation_losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title(plot_title)
    plt.text(0.02, 0.5, plot_text, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.3)
    plt.show()

    # TODO - Add a plot of accuracy.

    # TODO - Figure out how to create comparison plots of the different variants

    if verbose:
        print('Terminating...')


def main():
    """
    If you pass an argument only once with multiple variants it will be used in all applicable variants.
    By applicable we mean that the variant uses that parameter.

    For instance, it is not an error to not have a regularization coefficient if Regularized Gradient
    Descent was not specified and any other variant will ignore a regularization coefficient if one
    is supplied.

    Passing an argument multiple times without passing multiple variants is an error.

    The general rule of thumb is that if you have multiple variants then for any other argument
    except --verbose and --compare, the number of instances is at most one (if applicable to at least one of the
    variants) or equal to the number of variants\runs.

    Available variants include:

        GD - Gradient Descent - A vanilla Gradient Descent where we normalize the gradient.

        RGD - Regularized Gradient Descent - Adds a regularization term with a coefficient
              between 0 and 1 (inclusive).

        PGD - Projected Gradient Descent - Projects the weights to a hypersphere (ball) of
              radius R.

        CGD - Constrained Gradient Descent - Constraints each weight to be at most B
              (Projection to ball of radius R using the infinity-norm)

        SGD - Stochastic Gradient Descent - Gradient Descent where we perform the updates
              one sample at time instead of batches.

    Examples of valid input:

    "python main.py -v GD --epochs 50 --rate 0.3 --loss hinge --digits 0 9"

        This runs Gradient Descent with 50 epochs, a learning rate of 0.3 and with the hinge loss
        function learning to classify between 0 and 9.

    "python main.py -v GD --epochs 50 --rate 0.3 --loss hinge
                    -v GD --epochs 200 --rate 0.7 --digits 0 9
                    --compare"

        This will run two instances of Gradient Descent one with 50 epochs and a learning rate of 0.3 and
        one with 200 epochs and a learning rate of 0.7. Both instances will use the hinge loss function.
        and digits as only one of each specified. Lastly, the --compare option was specified so we will
        output a comparison graph between the two runs

    "python main.py -v GD --epochs 50 --rate 0.3 --loss hinge --digits 0 9
                   -v RGD --coefficient 0.5"

        This will run both a Gradient Descent and a Regularized Gradient Descent with the same parameters.
        Since the regularization coefficient is only applicable to Regularized Gradient Descent it can be
        specified only once.

    "python main.py -v SGD --epochs 50 --rate 0.3 --loss hinge --digits 8 0
                    -v PGD --epochs 100 --rate 0.1 --loss bce --radius 2 --digits 1 2
                    -v RGD --epochs 200 --rate 0.05 --loss hinge --coefficient 7.5 -- digits 7 3"

        This will run a Stochastic Gradient Descent, a Projected Gradient Descent and a Regularized Gradient
        Descent each with its own parameters. Notice that even though we already specified the hinge loss
        function for the Gradient Descent we still had to specify it for the Regularized Gradient Descent.
        That is, for any argument, once we pass more than once instance of it we have to pass an instance
        for every variant.

    Example of invalid input:"""
    args = parse_arguments()
    run_configurations = build_instances(args)


    for config in run_configurations:
        config.results = do_stuff(config)

        # TODO - Move the drawing of graphs to here. Maybe.
        plot_utils.draw_graph()

    # TODO - Compare the results when --compare is specified.
    plot_utils.draw_graph()

if __name__ == '__main__':
    main()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--digits', help='The classes for binary classification (e.g., "--digits 0 9" means [0, 9])', type=int, nargs=2, default=SELECTED_CLASSES)
    # parser.add_argument('-r', '--regularized', help='Use regularized gradient descent', type=float, default=None)
    # parser.add_argument('-p', '--projected', help='Use projected gradient descent', type=float, default=None)
    # parser.add_argument('-s', '--stochastic', help='Use stochastic gradient descent', action='store_true')
    # parser.add_argument('-l', '--loss', help='Choose the loss function to use.', type=str, required=True)
    #
    # parser.add_argument('--epochs', help='Number of epochs', type=int, default=NUM_EPOCHS)
    # parser.add_argument('--rate', help='Learning rate', type=float, default=LEARNING_RATE)
    #
    # parser.add_argument('-v', '--verbose', help='Prints extra information and details', action='store_true')
    #
    # args = parser.parse_args()
    #
    # main(args.rate, args.epochs, args.digits,
    #      args.regularized, args.stochastic, args.projected,
    #      args.loss, args.verbose)
