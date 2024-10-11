import matplotlib as plt

import gd


def plot_data_from_config(config : gd.GradientDescent, x_label, y_label, title, verbose=False):
    additional_info = build_plot_text(config)
    plot_data([(config.epochs, config.results.training_losses),
               (config.epochs, config.results.testing_losses)],
              x_label, y_label, title,
              additional_info=additional_info)
    if verbose:
        plot_data([(config.epochs, config.retuls.training_accuracy),
                   (config.epochs, config.retuls.testing_accuracy)],
                  x_label, y_label, title, additional_info=additional_info)

def plot_data_from_configs(configs : gd.RunConfiguration, x_label, y_label, title):
    for config in configs:
        plot_data_from_config(config, x_label, y_label, title)
        # additional_info = build_plot_text(config)
        # plot_data_from_config(config, x_label, y_label, title, verbose=configs.verbose)

        # if configs.verbose:
        #     plot_data([(config.epochs, config.results.testing_) for config in configs],
        #               [f'{gd.GD_VARIANT_MAPPING[type(config)]}' for config in configs],
        #               x_label, y_label, title, additional_info=additional_info)



def plot_data(data, x_label, y_label, title, legend=None, additional_info=None):
    for iterations, values in zip(data):
        plt.plot(range(iterations), values)
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


def build_plot_text(parameters):
    return build_plot_text(learning_rate=parameters.learning_rate,
                           selected_classes=parameters.digits,
                           number_of_epochs=parameters.epochs,
                           loss_function_name=parameters.loss_function,
                           regularization_coefficient=parameters.regularization_coefficient if parameters.regularization_coefficient is not None else '',
                           hypersphere_radius=parameters.radius if hasattr(parameters, 'radius') else None,
                           cutoff=parameters.cutoff if hasattr(parameters, 'cutoff') else None,
                           stochastic=type(parameters) is gd.StochasticGradientDescent,
                           train_loss=parameters.results.training_losses,
                           test_loss=parameters.results.testing_losses,
                           train_accuracy=parameters.results.training_accuracy,
                           test_accuracy=parameters.results.testing_accuracy)


def build_plot_text(learning_rate, selected_classes, number_of_epochs,
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