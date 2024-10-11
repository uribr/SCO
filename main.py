import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import gd
import gd_utils
import plot_utils
import main_utils
import hardcoded_config


class GDResults:
    def __init__(self):
        self.training_losses = []
        self.testing_losses = []

        self.training_accuracies = []
        self.testing_accuracies = []
        pass


def run(configs):
    if configs.verbose:
        print('Starting...')

    if configs.verbose:
        print('Loading the MNIST dataset...')

    if configs.verbose:
        print('Preprocessing...')

    np.random.seed(configs.seed)

    for config in configs:
        config.results = gradient_descent(config, configs.verbose)
        plot_utils.plot_data_from_config(config, configs.verbose)

    # TODO - Add text to the comparison plots to indicate learning rates, digits and special parameters of each curve.
    if configs.compare:
        plot_utils.plot_comparison(configs)
        plot_utils.plot_data([(config.epochs, config.results.testing_losses) for config in configs],
                             [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
                             'Iteration', 'Loss','Test Loss Comparison', '')
        plot_utils.plot_data([(config.epochs, config.results.testing_accuracies) for config in configs],
                             [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
                             'Iteration', 'Accuracy','Test Accuracy Comparison', '')

        if configs.verbose:
            # Verbosity adds a comparison of the training graphs.
            plot_utils.plot_data([(config.epochs, config.results.training_losses) for config in configs],
                                 [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
                                 'Iteration', 'Loss', 'Test Loss Comparison', '')
            plot_utils.plot_data([(config.epochs, config.results.training_accuracies) for config in configs],
                                 [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
                                 'Iteration', 'Accuracy', 'Test Accuracy Comparison', '')

    if configs.verbose:
        print('Terminating...')

def gradient_descent(parameters, verbose):
    # We'll just use the same variables for now.
    learning_rate = parameters.learning_rate
    number_of_epochs = parameters.epochs
    selected_classes = parameters.digits
    loss_function_name = parameters.loss_function
    stochastic = type(parameters) is gd.StochasticGradientDescent
    regularization_coefficient = None
    if hasattr(parameters, 'regularization_coefficient'):
        regularization_coefficient = parameters.regularization_coefficient
    hypersphere_radius = None
    if hasattr(parameters, 'radius'):
        hypersphere_radius = parameters.radius
    cutoff = None
    if hasattr(parameters, 'cutoff'):
        cutoff = parameters.cutoff

    gd_results = GDResults()

    # TODO - Move the loading and preprocessing outside of this function and do it once for all the runs unless there is
    #        some specialized preprocessing we would like to try on a specific variant.

    # Load and preprocess data
    mnist = fetch_openml('mnist_784')

    # Choose a loss function
    loss_function = None
    if str.lower(loss_function_name) == hardcoded_config.HINGE_LOSS_STRING:
        loss_function = gd_utils.hinge_loss
        labels = [-1, 1]
        pred_thr = 0
    elif str.lower(loss_function_name) == hardcoded_config.BCE_LOSS_STRING:
        loss_function = gd_utils.bce_loss
        labels = [0, 1]
        pred_thr = 0.5
    else:
        raise RuntimeError(f"Loss function {loss_function_name} is not supported")

    df = pd.DataFrame.from_dict(
        {'data': list(mnist['data'].astype('float64').values), 'target': mnist['target'].astype('int')})

    # filter by class, shuffle, divide to train/validation/test
    df = df.loc[df['target'].isin(selected_classes)]
    df['target'] = df['target'].replace(to_replace=selected_classes[0], value=labels[0])
    df['target'] = df['target'].replace(to_replace=selected_classes[1], value=labels[1])

    df = df.sample(frac=1).reset_index(drop=True)

    # train/validation/test
    data_split = [hardcoded_config.TRAINING_SET_RELATIVE_SIZE,
                  hardcoded_config.TEST_SET_RELATIVE_SIZE]

    num_samples = len(df)
    df_train = df.iloc[:int(num_samples * data_split[0]), :]
    df_test = df.iloc[int(num_samples * data_split[0]): int(num_samples * (data_split[1] + data_split[0])), :]

    train_data = np.stack(df_train['data'].to_numpy())
    test_data = np.stack(df_test['data'].to_numpy())

    # Adding a bias term
    train_data = np.concatenate((train_data, np.ones((len(train_data), 1), dtype='float64')), axis=1)
    test_data = np.concatenate((test_data, np.ones((len(test_data), 1), dtype='float64')), axis=1)

    train_data = train_data / 255.
    test_data = test_data / 255.

    train_targets = df_train['target'].to_numpy()
    test_targets = df_test['target'].to_numpy()

    # TODO - Most of what comes before this point is preprocessing and should be moved out of this function. The real GD starts here.

    # model
    # initialize weights
    weights = np.random.randn(hardcoded_config.WEIGHT_LENGTH) * np.sqrt(1 / hardcoded_config.WEIGHT_LENGTH)
    weights[-1] = 0

    # For plotting
    training_losses = []
    test_losses = []
    training_accuracies = []
    test_accuracies = []

    # Training loop
    epoch_loss = 0
    use_projection = hypersphere_radius is not None
    use_regularization = regularization_coefficient is not None
    use_clipping = cutoff is not None
    for epoch in range(number_of_epochs):
        if stochastic:
            for i in range(len(train_data)):
                sample_loss, grad = loss_function(
                    np.expand_dims(train_targets[i], 0), np.expand_dims(train_data[i], 0), weights)
                epoch_loss += sample_loss
                weights = gd_utils.update_weights_vanilla(weights, grad, learning_rate)
            epoch_loss = epoch_loss / len(train_data)
            test_loss, _ = loss_function(test_targets, test_data, weights)

        else:

            epoch_loss, grads = loss_function(train_targets, train_data, weights)

            if use_regularization:
                grads += 2 * regularization_coefficient * weights

            new_weights = gd_utils.update_weights_vanilla(weights, grads, learning_rate)

            if use_projection:
                if np.linalg.norm(new_weights) > hypersphere_radius:
                    new_weights = new_weights * hypersphere_radius / np.linalg.norm(new_weights)
            if use_clipping:
                new_weights = np.clip(new_weights, -cutoff, cutoff)

            weights = new_weights

            if verbose:
                if (use_regularization or use_projection) and epoch % hardcoded_config.REPORT_FREQUENCY == 0:
                    print(f'Weights Norm: {np.linalg.norm(weights)}')

        # Compute and store losses
        training_losses.append(epoch_loss)

        test_epoch_loss, _ = loss_function(test_targets, test_data, weights)
        test_losses.append(test_epoch_loss)

        # Compute and store accuracies
        train_y_pred = np.dot(train_data, weights.transpose())
        test_y_pred = np.dot(test_data, weights.transpose())

        if loss_function_name == hardcoded_config.BCE_LOSS_STRING:
            train_y_pred = gd_utils.sigmoid(train_y_pred)
            test_y_pred = gd_utils.sigmoid(test_y_pred)

        training_accuracy = gd_utils.binary_accuracy(train_y_pred, train_targets, pred_thr, labels) * 100
        training_accuracies.append(training_accuracy)

        test_accuracy = gd_utils.binary_accuracy(test_y_pred, test_targets, pred_thr, labels) * 100
        test_accuracies.append(test_accuracy)

    print(f'Train Accuracy: {training_accuracies[-1]:.3f} %, validation Accuracy: {test_accuracies[-1]:.3f} %\n')
    print(f'Train Loss: {training_losses[-1]:.3f}, Validation Loss: {test_losses[-1]:.3f}\n')

    gd_results.training_losses = training_losses
    gd_results.testing_losses = test_losses
    gd_results.training_accuracies = training_accuracies
    gd_results.testing_accuracies = test_accuracies

    # # Format title
    # plot_title = "Loss vs. Iterations"
    #
    # # Format extra information
    # plot_text = plot_utils.build_plot_text(learning_rate, selected_classes, number_of_epochs,
    #                                        loss_function_name, regularization_coefficient,
    #                                        hypersphere_radius, stochastic, training_losses[-1],
    #                                        test_losses[-1], training_accuracies[-1],
    #                                        test_accuracies[-1])
    #
    # plt.plot(range(number_of_epochs), training_losses)
    # plt.plot(range(number_of_epochs), test_losses)
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend(['Train Loss', 'Test Loss'])
    # plt.title(plot_title)
    # plt.text(0.02, 0.5, plot_text, transform=plt.gcf().transFigure)
    # plt.subplots_adjust(left=0.3)
    # plt.show()

    return gd_results



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
    parser = main_utils.GDArgumentParser()
    args = parser.parse_arguments()
    run_configurations = parser.build_instances(args)

    run(run_configurations)

if __name__ == '__main__':
    main()
