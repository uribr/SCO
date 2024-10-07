import sys

import unittest

import hardcoded_config
import main_utils
import gd_utils
import gd

class TestArgumentParsing(unittest.TestCase):
    def test_example_no_1(self):
        # Setting the command-line
        sys.argv = ['main.py', '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9']
        parser = main_utils.GDArgumentParser()
        args = parser.parse_arguments()

        self.assertEqual(args.compare, False)
        self.assertEqual(args.verbose, False)
        self.assertEqual(len(args.variants), 1)

        self.assertEqual(args.variants[0], gd.GradientDescentVariant.GD.name)
        self.assertEqual(args.epochs[0], '50')
        self.assertEqual(args.learning_rates[0], '0.3')
        self.assertEqual(args.loss_functions[0], hardcoded_config.HINGE_LOSS_STRING)
        self.assertEqual(args.digits[0], [0, 9])

    def test_example_no_2(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge',
                    '-v', 'GD', '--epochs', '200', '--rate', '0.7', '--digits', '0', '9',
                    '--compare']
        parser = main_utils.GDArgumentParser()
        args = parser.parse_arguments()

        self.assertEqual(args.verbose, False)
        self.assertEqual(args.compare, True)
        self.assertEqual(len(args.variants), 2)

        self.assertEqual(args.variants[0], gd.GradientDescentVariant.GD.name)
        self.assertEqual(args.epochs[0], '50')
        self.assertEqual(args.learning_rates[0], '0.3')
        self.assertEqual(args.loss_functions[0], hardcoded_config.HINGE_LOSS_STRING)
        self.assertEqual(args.digits[0], [0, 9])

        self.assertEqual(args.variants[1], gd.GradientDescentVariant.GD.name)
        self.assertEqual(args.epochs[1], '200')
        self.assertEqual(args.learning_rates[1], '0.7')


    def test_example_no_3(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9',
                    '-v', 'RGD', '--coefficient', '0.5', '--verbose']
        parser = main_utils.GDArgumentParser()
        args = parser.parse_arguments()

        self.assertEqual(args.compare, False)
        self.assertEqual(args.verbose, True)
        self.assertEqual(len(args.variants), 2)

        self.assertEqual(args.variants[0], gd.GradientDescentVariant.GD.name)
        self.assertEqual(args.epochs[0], '50')
        self.assertEqual(args.learning_rates[0], '0.3')
        self.assertEqual(args.loss_functions[0], hardcoded_config.HINGE_LOSS_STRING)
        self.assertEqual(args.digits[0], [0, 9])

        self.assertEqual(args.variants[1], gd.GradientDescentVariant.RGD.name)
        self.assertEqual(args.coefficients[0], '0.5')

    def test_example_no_4(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'SGD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '8', '0',
                    '-v', 'PGD', '--epochs', '100', '--rate', '0.1', '--loss', 'bce', '--radius', '2', '--digits', '1', '2',
                    '-v', 'RGD', '--epochs', '200', '--rate', '0.05', '--loss', 'hinge', '--coefficient', '7.5', '--digits', '7', '3']

        parser = main_utils.GDArgumentParser()
        args = parser.parse_arguments()

        self.assertEqual(args.compare, False)
        self.assertEqual(args.verbose, False)
        self.assertEqual(len(args.variants), 3)

        self.assertEqual(args.variants[0], gd.GradientDescentVariant.SGD.name)
        self.assertEqual(args.epochs[0], '50')
        self.assertEqual(args.learning_rates[0], '0.3')
        self.assertEqual(args.loss_functions[0], hardcoded_config.HINGE_LOSS_STRING)
        self.assertEqual(args.digits[0], [8, 0])

        self.assertEqual(args.variants[1], gd.GradientDescentVariant.PGD.name)
        self.assertEqual(args.epochs[1], '100')
        self.assertEqual(args.learning_rates[1], '0.1')
        self.assertEqual(args.radii[0], '2')
        self.assertEqual(args.loss_functions[1], hardcoded_config.BCE_LOSS_STRING)
        self.assertEqual(args.digits[1], [1, 2])

        self.assertEqual(args.variants[2], gd.GradientDescentVariant.RGD.name)
        self.assertEqual(args.epochs[2], '200')
        self.assertEqual(args.learning_rates[2], '0.05')
        self.assertEqual(args.coefficients[0], '7.5')
        self.assertEqual(args.loss_functions[2], hardcoded_config.HINGE_LOSS_STRING)
        self.assertEqual(args.digits[2], [7, 3])

    def test_invalid_input_no_1(self):
        # Setting the command-line
        sys.argv = ['main.py', '-v', 'GD', '--epochs', '70', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9']
        parser = main_utils.GDArgumentParser()
        with self.assertRaises(expected_exception=RuntimeError, msg='The number of variants does not match the number of epochs supplied.'):
            parser.parse_arguments()

    def test_invalid_input_no_2(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge',
                    '-v', 'GD', '--epochs', '200', '--rate', '0.7', '--digits', '0', '9',
                    '-v', 'GD', '--epochs', '200', '--digits', '0', '9',
                    '--compare']
        parser = main_utils.GDArgumentParser()
        with self.assertRaises(expected_exception=RuntimeError, msg='The number of variants does not match the number of learning rates supplied.'):
            parser.parse_arguments()

class TestBuildInstances(unittest.TestCase):
    def test_invalid_input_no_1(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'NoGD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9',
                    '-v', 'RGD', '--coefficient', '0.5']
        parser = main_utils.GDArgumentParser()
        with self.assertRaises(expected_exception=RuntimeError, msg='Unknown variant "NoGD"'):
            parser.build_instances(parser.parse_arguments())

if __name__ == '__main__':
    unittest.main()