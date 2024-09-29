
def build_plot_text(learning_rate, selected_classes, number_of_epochs,
                    loss_function_name, regularization_coefficient,
                    hypersphere_radius, stochastic, train_accuracy,
                    validation_accuracy):
    plot_text = f'Digits: {selected_classes[0]}, {selected_classes[-1]}\n'\
                f'Rate: {learning_rate}\n'\
                f'Iterations: {number_of_epochs}\n'\
                f'Loss: {loss_function_name}\n'\
                f'Train Acc: {train_accuracy:.2f}\n'\
                f'Validation Acc: {validation_accuracy:.2f}\n'\
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
