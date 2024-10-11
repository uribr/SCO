import matplotlib as plt

import gd


def plot_data_from_config(config, x_label, y_label, title, verbose=False):
    additional_info = build_plot_text(config)
    plot_data([config.epochs, config.epochs],
               [config.results.training_losses, config.results.testing_losses],
              x_label, y_label, title, legend=['train', 'test'], additional_info=additional_info)
    if verbose:
        plot_data([config.epochs, config.epochs],
                   [config.results.training_accuracy, config.results.testing_accuracy],
                  x_label, y_label, title, legend=['train', 'test'], additional_info=additional_info)


def plot_data_from_configs(configs: gd.RunConfiguration, x_label, y_label, title):
    for config in configs:
        plot_data_from_config(config, x_label, y_label, title, configs.verbose)
        # additional_info = build_plot_text(config)
        # plot_data_from_config(config, x_label, y_label, title, verbose=configs.verbose)

        # if configs.verbose:
        #     plot_data([(config.epochs, config.results.testing_) for config in configs],
        #               [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
        #               x_label, y_label, title, additional_info=additional_info)


def plot_data(epochs, values, x_label, y_label, title, legend=None, additional_info=None):
    for _epochs, _values in zip(epochs, values):
        plt.plot(range(_epochs), _values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if legend is not None:
        plt.legend(legend)

    if additional_info is not None:
        plt.text(0.02, 0.5,
                 additional_info,
                 transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.3)

    plt.show()


def plot_comparison(configs):
    pass
    # plot_data([(config.epochs, config.results.testing_losses) for config in configs],
    #           x_label, y_label, title,
    #           legend=[f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
    #           additional_info=additional_info)


def build_legend_text(configs):
    return [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs]


def _build_plot_text(learning_rate: object, selected_classes: object, number_of_epochs: object,
                     loss_function_name: object, regularization_coefficient: object,
                     hypersphere_radius: object, cutoff: object, stochastic: object, train_loss: object,
                     test_loss: object, train_accuracy: object, test_accuracy: object) -> object:
    plot_text = f'Digits: {selected_classes[0]}, {selected_classes[-1]}\n' \
                f'Rate: {learning_rate}\n' \
                f'Iterations: {number_of_epochs}\n' \
                f'Loss: {loss_function_name}\n' \
                f'Train Loss: {train_loss:.3f}\n' \
                f'Test Loss: {test_loss:.3f}\n' \
                f'Train Acc: {train_accuracy:.3f}\n' \
                f'Test Acc: {test_accuracy:.3f}\n' \
                'Variant: '
    if regularization_coefficient is not None:
        plot_text += 'RGD\n'
        plot_text += f'Coefficient: {regularization_coefficient}\n'
    elif hypersphere_radius is not None:
        plot_text += 'PGD\n'
        plot_text += f'Radius: {hypersphere_radius}\n'
    elif stochastic:
        plot_text += 'SGD\n'
    elif cutoff is not None:
        plot_text += 'CGD\n'
        plot_text += f'Cutoff: {cutoff}\n'
    else:
        plot_text += 'GD\n'

    return plot_text


def build_plot_text(parameters):
    # coefficient = None
    # if hasattr(parameters, 'regularization_coefficient'):
    #     coefficient = parameters.regularization_coefficient
    #
    # radius = None
    # if hasattr(parameters, 'hypersphere_radius'):
    #     radius = parameters.hypersphere_radius
    #
    # cutoff = None
    # if hasattr(parameters, 'cutoff'):
    #     cutoff = parameters.cutoff

    return _build_plot_text(learning_rate=parameters.learning_rate,
                            selected_classes=parameters.digits,
                            number_of_epochs=parameters.epochs,
                            loss_function_name=parameters.loss_function,
                            regularization_coefficient=parameters.regularization_coefficient if hasattr(parameters, 'regularization_coefficient') else None,
                            hypersphere_radius=parameters.hypersphere_radius if hasattr(parameters, 'hypersphere_radius') else None,
                            cutoff=parameters.cutoff if hasattr(parameters, 'cutoff') else None,
                            stochastic=type(parameters) is gd.StochasticGradientDescent,
                            train_loss=parameters.results.training_losses[-1],
                            test_loss=parameters.results.testing_losses[-1],
                            train_accuracy=parameters.results.training_accuracies[-1],
                            test_accuracy=parameters.results.testing_accuracies[-1])
