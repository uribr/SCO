import argparse

import gd
import hardcoded_config

class GDArgumentParser:
    def __init__(self):
        self.number_of_variants = 0
        self.number_of_epochs = 0
        self.number_of_digits = 0
        self.number_of_learning_rates = 0
        self.number_of_loss_functions = 0
        self.number_of_radii = 0
        self.number_of_cutoffs = 0
        self.number_of_coefficients = 0

        self.count_gd = 0
        self.rgd_count = 0
        self.count_sgd = 0
        self.pgd_count = 0
        self.cgd_count = 0

        self.initialized = True

    def _initialize(self):
        self.number_of_variants = 0
        self.number_of_epochs = 0
        self.number_of_digits = 0
        self.number_of_learning_rates = 0
        self.number_of_loss_functions = 0
        self.number_of_radii = 0
        self.number_of_cutoffs = 0
        self.number_of_coefficients = 0

        self.count_gd = 0
        self.rgd_count = 0
        self.count_sgd = 0
        self.pgd_count = 0
        self.cgd_count = 0

    def parse_arguments(self):
        self._initialize()

        parser = argparse.ArgumentParser(description='Runs (possibly multiple) instances of Gradient Descent and its variants.'
                                                     'Arguments can be passed more than once so we can run multiple variants '
                                                     'with different parameters.')

        parser.add_argument('-s', '--seed', dest='seed', type=int, default=1337,
                            help='Seed for the random number generator.')

        parser.add_argument('-v', '--variant', dest='variants', type=str, action='append', required=True,
                            help='The names of the Gradient Descent variants to run.'
                                 '-v and\\or --variant can be specified multiple times. However, for each instance of variant'
                                 'you must provide a set of parameters so the number of all the arguments is the same.\n'
                                 'The only exceptions is that if you want all the runs to use the same value for one of the '
                                 'arguments then you can specify that argument only once.')

        parser.add_argument('-d', '--digits', type=int, nargs=2, action='append', required=True,
                            help='The classes for binary classification (e.g., "--digits 0 9" means [(0, 9)] '
                                 'and "--digits 0 9 1 2" means [(0,9), (1,2)]). Can be specified multiple times.')


        parser.add_argument('--loss', '--loss_function', dest='loss_functions', default=[], action='append',
                            type=str, required=True,
                            help='The loss function to use for optimization during Gradient Descent. Can be specified '
                                 'multiple times.')

        parser.add_argument('--epochs', default=[], action='append', required=True,
                            help='Number of epochs (iterations) to run the Gradient Descent variant. '
                                 'Can be specified multiple times.')

        parser.add_argument('--rate', '--learning_rate', dest='learning_rates', default=[], action='append',
                            required=True,
                            help='The learning rate to use when updating the weights during Gradient Descent variant. '
                                 'Can be specified multiple times.')

        parser.add_argument('--coefficient', dest='coefficients', default=[], action='append',
                            help='The regularization coefficient to be used. Can be specified multiple times '
                                 '(see help about variants and examples above)')

        parser.add_argument('--radius', dest='radii', default=[], action='append',
                            help='The radius of the hypersphere\\ball that the weights will be projected into when running '
                                 'a Projected Gradient Descent. Can be specified multiple times.')

        parser.add_argument('--cutoff', dest='cutoffs', default=[], action='append',
                            help='The cutoff value to clip the values of the weight vector when running Constrained '
                                 'Gradient Descent. Can be specified multiple times.')

        parser.add_argument("--compare", action='store_true',
                            help='Generate additional graphs for comparing the loss and accuracy against the number of '
                                 'iterations between all the Gradient Descent variants provided. May only be specified '
                                 'once.')

        parser.add_argument('--verbose', action='store_true',
                            help='Prints additional information and details. May only be specified once.')

        args = parser.parse_args()

        self._validate_arguments(args)

        self.initialized = False

        return args

    def _validate_arguments(self, args):
        for variant in args.variants:
            match variant:
                case gd.GradientDescentVariant.GD.name:
                    self.count_gd += 1
                case gd.GradientDescentVariant.RGD.name:
                    self.rgd_count += 1
                case gd.GradientDescentVariant.PGD.name:
                    self.pgd_count += 1
                case gd.GradientDescentVariant.CGD.name:
                    self.cgd_count += 1
                case gd.GradientDescentVariant.SGD.name:
                    self.count_sgd += 1

        self.number_of_variants = len(args.variants)
        self.number_of_epochs = len(args.epochs)
        self.number_of_digits = len(args.digits)
        self.number_of_learning_rates = len(args.learning_rates)
        self.number_of_loss_functions = len(args.loss_functions)
        self.number_of_radii = len(args.radii)
        self.number_of_cutoffs = len(args.cutoffs)
        self.number_of_coefficients = len(args.coefficients)

        # The number of instances of a parameter matches the number of instances to run if they
        # are equal (this includes the case where they are both 0) or if the number of times the parameter appears is 1.
        if not self._verify_variant_count_matches_paramerter_count(self.number_of_variants, self.number_of_learning_rates):
            raise RuntimeError('The number of variants does not match the number of learning rates supplied.')
        if not self._verify_variant_count_matches_paramerter_count(self.number_of_variants, self.number_of_loss_functions):
            raise RuntimeError('The number of variants does not match the number of loss functions supplied.')
        if not self._verify_variant_count_matches_paramerter_count(self.number_of_variants, self.number_of_epochs):
            raise RuntimeError('The number of variants does not match the number of epochs supplied.')
        if not self._verify_variant_count_matches_paramerter_count(self.number_of_variants, self.number_of_digits):
            raise RuntimeError('The number of variants does not match the number of digits supplied.')

        # The number of instances of a variant's parameter matches the number of instances of that variant if they
        # are equal (this includes the case where they are both 0) or if the number of times the parameter appears
        # is 1 and the number of instances of the variant > 0
        if not self._verify_variant_count_matches_paramerter_count(self.pgd_count, self.number_of_radii):
            raise RuntimeError('The number of PGD instances does not match the number of radii supplied.')
        if not self._verify_variant_count_matches_paramerter_count(self.cgd_count, self.number_of_cutoffs):
            raise RuntimeError('The number of CGD instances does not match the number of cutoffs supplied.')
        if not self._verify_variant_count_matches_paramerter_count(self.rgd_count, self.number_of_coefficients):
            raise RuntimeError('The number of RGD instances does not match the number of coefficients supplied.')

    @staticmethod
    def _verify_variant_count_matches_paramerter_count(variant_count, parameter_count):
        is_valid = False

        # The input is valid if there is a parameter for every instance
        # of the variant to run or there is a one for all of them.
        if parameter_count == variant_count:
            # This is the case where each instance of the variant has its own parameter
            is_valid = True
        elif parameter_count > 0 and variant_count > 0:
            # This is the case where all the instances of hte variant use the same parameter.
            is_valid = parameter_count == 1
        else:
            # Either the variant was specified and the parameter not or the other way around.
            # Leave the default value of False.
            pass

        return is_valid

    @staticmethod
    def _pad_parameter_list_to_length(original_parameter_list, desired_length):
        assert(len(original_parameter_list) in (0, 1, desired_length))

        length = len(original_parameter_list)
        if length == 0:
            original_parameter_list.extend(None for i in range(desired_length))
        elif length == 1:
            tmp = [original_parameter_list[0] for i in range(desired_length - len(original_parameter_list))]
            original_parameter_list.extend(tmp)
        elif length == desired_length:
            # Nothing to do in this case.
            pass
        else:
            # Shouldn't get here
            raise RuntimeError('Got an invalid number of parameters post validation.')

    def _pad_args_to_length(self, args):
        self._pad_parameter_list_to_length(args.digits, self.number_of_variants)
        self._pad_parameter_list_to_length(args.learning_rates, self.number_of_variants)
        self._pad_parameter_list_to_length(args.loss_functions, self.number_of_variants)
        self._pad_parameter_list_to_length(args.epochs, self.number_of_variants)
        self._pad_parameter_list_to_length(args.coefficients, self.number_of_variants)
        self._pad_parameter_list_to_length(args.cutoffs, self.number_of_variants)
        self._pad_parameter_list_to_length(args.radii, self.number_of_variants)


    def build_instances(self, args):
        self._pad_args_to_length(args)
        configs = gd.RunConfiguration(args.seed, args.verbose, args.compare)

        for i in range(self.number_of_variants):
            variant = args.variants[i]
            match variant:
                case gd.GradientDescentVariant.GD.name:
                    configs.append(gd.GradientDescent(args.digits[i],
                                                      float(args.learning_rates[i]),
                                                      int(args.epochs[i]),
                                                      args.loss_functions[i]))
                case gd.GradientDescentVariant.RGD.name:
                    configs.append(gd.RegularizedGradientDescent(args.digits[i],
                                                                 float(args.learning_rates[i]),
                                                                 int(args.epochs[i]),
                                                                 args.loss_functions[i],
                                                                 float(args.coefficients[i])))
                case gd.GradientDescentVariant.PGD.name:
                    configs.append(gd.ProjectedGradientDescent(args.digits[i],
                                                               float(args.learning_rates[i]),
                                                               int(args.epochs[i]),
                                                               args.loss_functions[i],
                                                               float(args.radii[i])))
                case gd.GradientDescentVariant.CGD.name:
                    configs.append(gd.ConstrainedGradientDescent(args.digits[i],
                                                                 float(args.learning_rates[i]),
                                                                 int(args.epochs[i]),
                                                                 args.loss_functions[i],
                                                                 float(args.cutoffs[i])))
                case gd.GradientDescentVariant.SGD.name:
                    configs.append(gd.StochasticGradientDescent(args.digits[i],
                                                                float(args.learning_rates[i]),
                                                                int(args.epochs[i]),
                                                                args.loss_functions[i]))
                case _:
                    raise RuntimeError(f'Unknown variant "{variant}"')

        return configs