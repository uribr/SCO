from enum import Enum

from gd_utils import *

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
    def __init__(self, digits, learning_rate, iterations, variant = GradientDescentVariant.GD):
        self.digits = digits
        self.variant = variant
        self.iterations = iterations
        self.learning_rate = learning_rate

    def __call__(self):
        pass

class StochasticGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, iterations):
        super(StochasticGradientDescent, self).__init__(digits, learning_rate, iterations, GradientDescentVariant.SGD)


class RegularizedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, iterations, regularization_coefficient):
        super(RegularizedGradientDescent, self).__init__(digits, learning_rate, iterations, GradientDescentVariant.RGD)
        self.regularization_coefficient = regularization_coefficient


class ProjectedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, iterations, radius):
        super(ProjectedGradientDescent, self).__init__(digits, learning_rate, iterations, GradientDescentVariant.PGD)
        self.radius = radius


class ConstrainedGradientDescent(GradientDescent):
    def __init__(self, digits, learning_rate, iterations, cutoff_value):
        super(ConstrainedGradientDescent, self).__init__(digits, learning_rate, iterations, GradientDescentVariant.CGD)
        self.cutoff_value = cutoff_value
