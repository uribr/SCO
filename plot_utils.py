import matplotlib.pyplot as plt

import gd


def plot_data_from_config(config, verbose=False):
    additional_info = build_plot_text(config)
    plot_data([config.epochs, config.epochs],
              [config.results.training_losses, config.results.testing_losses],
              'Loss', 'Epoch', 'Loss vs. Epoch',
              legend=['train', 'test'], additional_info=additional_info)

    plot_data([config.epochs, config.epochs],
              [config.results.training_accuracies, config.results.testing_accuracies],
              'Accuracy', 'Epoch', 'Accuracy vs. Epoch',
              legend=['train', 'test'], additional_info=additional_info)


def plot_data_from_configs(configs: gd.RunConfiguration, x_label, y_label, title):
    for config in configs:
        plot_data_from_config(config, configs.verbose)


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
    shared_params = _find_shared_parameters(configs)

    additional_loss_info = build_comparison_plot_text(configs, shared_params)
    loss_legend = build_loss_legend_text(configs, shared_params)
    plot_data([config.epochs for config in configs],
              [config.results.testing_losses for config in configs],
              'Iteration', 'Loss', 'Test Loss Comparison',
              legend= loss_legend, additional_info=additional_loss_info)

    additional_acc_info = build_comparison_plot_text(configs, shared_params)
    acc_legend = build_accuracy_legend_text(configs, shared_params)
    plot_data([config.epochs for config in configs],
              [config.results.testing_accuracies for config in configs],
              'Iteration', 'Loss', 'Test Accuracy Comparison',
              legend= acc_legend, additional_info=additional_acc_info)

    if configs.verbose:
        # Verbosity adds a comparison of the training graphs.
        plot_data([config.epochs for config in configs],
                  [config.results.training_losses for config in configs],
                  'Iteration', 'Loss', 'Training Loss Comparison',
                  legend=loss_legend, additional_info=additional_loss_info)
        plot_data([config.epochs for config in configs],
                  [config.results.training_accuracies for config in configs],
                  'Iteration', 'Loss', 'Training Accuracy Comparison',
                  legend= acc_legend, additional_info=additional_acc_info)



def _build_common_legend_text(params, config, verbose):
    variant_text = f'{gd.GD_VARIANT_MAPPING[type(config)]}'
    if params.digits is None:
        variant_text += f'[{config.digits[0]}, {config.digits[1]}]'
    if params.epochs is None and verbose:
        variant_text += f'|E: {config.epochs}'
    if params.rate is None:
        variant_text += f'|R: {config.learning_rate:.1f}'
    if params.loss is None:
        variant_text += f'|L: {config.loss_function}'

    if params.cutoff is None and isinstance(config, gd.ConstrainedGradientDescent):
        variant_text += f'|P:{config.cutoff_value:.1f}'
    if params.coefficient is None and isinstance(config, gd.RegularizedGradientDescent):
        variant_text += f'|P:{config.regularization_coefficient:.1f}'
    if params.radius is None and isinstance(config, gd.ProjectedGradientDescent):
        variant_text += f'|P:{config.radius:.1f}'

    return variant_text

def build_loss_legend_text(configs, params):
    legend = []

    for config in configs:
        variant_text = _build_common_legend_text(params, config, configs.verbose)
        variant_text += f'|TS: {config.results.testing_losses[-1]:.1f}'

        if configs.verbose:
            variant_text += f'|TR: {config.results.training_losses[-1]:.1f}'

        legend.append(variant_text)
    return legend

def build_accuracy_legend_text(configs, params):
    legend = []

    for config in configs:
        variant_text = _build_common_legend_text(params, config, configs.verbose)
        variant_text += f'|TS: {config.results.testing_accuracies[-1]:.1f}'

        if configs.verbose:
            variant_text += f'|TR: {config.results.training_accuracies[-1]:.1f}'

        legend.append(variant_text)
    return legend

def _build_plot_text(learning_rate, selected_classes, number_of_epochs,
                     loss_function_name, regularization_coefficient,
                     hypersphere_radius, cutoff, stochastic, train_loss,
                     test_loss, train_accuracy, test_accuracy):
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

class Parameters:
    def __init__(self, digits=None, epochs=None, rate=None, loss=None, coefficient=None, radius=None, cutoff=None):
        self.digits = digits
        self.epochs = epochs
        self.rate = rate
        self.loss = loss
        self.coefficient = coefficient
        self.radius = radius
        self.cutoff = cutoff

def _sort(lst: list):
    lst.sort()
    return lst

def _find_shared_parameters(configs):
    params = Parameters()

    digits = {tuple(_sort(config.digits)) for config in configs}
    epochs = {config.epochs for config in configs}
    rates = {config.learning_rate for config in configs}
    loss = {config.loss_function for config in configs}

    coefficients = {config.regularization_coefficient if hasattr(config, 'regularization_coefficient') else None for
                    config in configs}
    if None in coefficients:
        coefficients.remove(None)

    radii = {config.radius if hasattr(config, 'radius') else None for config in configs}
    if None in radii:
        radii.remove(None)

    cutoffs = {config.cutoff_value if hasattr(config, 'cutoff_value') else None for config in configs}
    if None in cutoffs:
        cutoffs.remove(None)

    if len(digits) == 1:
        params.digits = digits.pop()

    if len(rates) == 1:
        params.rate = rates.pop()

    if len(epochs) == 1:
        params.epochs = epochs.pop()

    if len(loss) == 1:
        params.loss = loss.pop()

    if len(coefficients) == 1:
        params.coefficient = coefficients.pop()

    if len(cutoffs) == 1:
        params.cutoff = cutoffs.pop()

    if len(radii) == 1:
        params.radius = radii.pop()

    return params

def build_comparison_plot_text(configs, params):
    """
    The parameters that will show up are only those that are shared across all the configurations
    where they are applicable. That is, if all the RGD configurations have the same coefficient
    it will show even if there are some configurations which aren't RGD.
    :param configs: RunConfigurations
    :return:
    """
    plot_text =''

    if params.digits is not None:
        plot_text += f'Digits: {params.digits[0]}, {params.digits[-1]}\n'

    if params.rate is not None:
        plot_text += f'Rate: {params.rate}\n'

    if params.epochs is not None:
        plot_text += f'Iterations: {params.epochs}\n'

    if params.loss is not None:
        plot_text += f'Loss: {params.loss}\n'

    if params.coefficient is not None:
        plot_text += f'Coefficient: {params.coefficient}\n'

    if params.cutoff is not None:
        plot_text += f'Cutoff: {params.cutoff}\n'

    if params.radius is not None:
        plot_text += f'Radius: {params.radius}\n'

    return plot_text


def build_plot_text(parameters):
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
