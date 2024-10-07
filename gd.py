from enum import Enum

from gd_utils import *

# To be honest... If I had a system where the number of variants might increase, which isn't really farfetched for
# learning in my opinion, I would've used a dictionary instead of classes to represent each variant because it would
# be a lot easier to add new ones.
# I'd use enum of strings or something like that for easy to use keys and each run would be a dictionary.

class GradientDescentVariant(Enum):
    GD = "GD"   # Gradient Descent
    RGD = "RGD" # Regularized Gradient Descent
    PGD = "PGD" # Projected Gradient Descent
    CGD = "CGD" # Constrained Gradient Descent
    SGD = "SGD" # Stochastic Gradient Descent

class RunConfiguration(list):
    def __init__(self, verbose, compare, seq=()):
        self.verbose = verbose
        self.compare = compare
        super(RunConfiguration, self).__init__(seq)



class GradientDescent:
    def __init__(self, digits, learning_rate, epochs, loss_function):
        self.digits = digits
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function


class RegularizedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, epochs, loss_function, regularization_coefficient):
        super(RegularizedGradientDescent, self).__init__(digits, learning_rate, epochs, loss_function)
        self.regularization_coefficient = regularization_coefficient


class ProjectedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, epochs, loss_function, radius):
        super(ProjectedGradientDescent, self).__init__(digits, learning_rate, epochs, loss_function)
        self.radius = radius


class ConstrainedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, epochs, loss_function, cutoff_value):
        super(ConstrainedGradientDescent, self).__init__(digits, learning_rate, epochs, loss_function)
        self.cutoff_value = cutoff_value


class StochasticGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, epochs, loss_function):
        super(StochasticGradientDescent, self).__init__(digits, learning_rate, epochs, loss_function)