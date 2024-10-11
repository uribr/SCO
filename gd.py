from enum import Enum

from gd_utils import *

class GradientDescentVariant(Enum):
    GD = "GD"  # Gradient Descent
    RGD = "RGD"  # Regularized Gradient Descent
    PGD = "PGD"  # Projected Gradient Descent
    CGD = "CGD"  # Constrained Gradient Descent
    SGD = "SGD"  # Stochastic Gradient Descent


class RunConfiguration(list):
    def __init__(self, seed, verbose, compare, seq=()):
        self.seed = seed
        self.verbose = verbose
        self.compare = compare
        super(RunConfiguration, self).__init__(seq)


class GradientDescent:
    def __init__(self, digits, learning_rate, epochs, loss_function):
        self.digits = digits
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.results = None


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


GD_VARIANT_MAPPING = {GradientDescent: GradientDescentVariant.GD.name,
                      RegularizedGradientDescent: GradientDescentVariant.RGD.name,
                      ProjectedGradientDescent: GradientDescentVariant.PGD.name,
                      ConstrainedGradientDescent: GradientDescentVariant.CGD.name,
                      StochasticGradientDescent: GradientDescentVariant.SGD.name}
