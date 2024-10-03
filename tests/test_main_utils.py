import sys

import unittest

import main_utils
import gd_utils
import gd

class TestArgumentParsing(unittest.TestCase):
    def test_example_no_1(self):
        # Setting the command-line
        sys.argv = ['main.py', '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9']
        configs = main_utils.parse_arguments()

        self.assertEqual(len(configs), 1)

        self.assertEqual(configs[0].variant, gd.GradientDescentVariant.GD)
        self.assertEqual(configs[0].epochs, 50)
        self.assertEqual(configs[0].learning_rate, 0.3)
        self.assertEqual(configs[0].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[0].classes, [0, 9])

    def test_example_no_2(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge',
                    '-v', 'GD', '--epochs', '200', '--rate', '0.7', '--digits', '0', '9',
                    '--compare']
        configs = main_utils.parse_arguments()

        self.assertEqual(len(configs), 2)
        self.assertEqual(configs.compare, True)

        self.assertEqual(configs[0].variant, gd.GradientDescentVariant.GD)
        self.assertEqual(configs[0].epochs, 50)
        self.assertEqual(configs[0].learning_rate, 0.3)
        self.assertEqual(configs[0].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[0].classes, [0, 9])

        self.assertEqual(configs[1].variant, gd.GradientDescentVariant.GD)
        self.assertEqual(configs[1].epochs, 200)
        self.assertEqual(configs[1].learning_rate, 0.7)
        self.assertEqual(configs[1].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[1].classes, [0, 9])


    def test_example_no_3(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'GD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '0', '9',
                    '-v', 'RGD', '--coefficient', '0.5']
        configs = main_utils.parse_arguments()

        self.assertEqual(len(configs), 2)

        self.assertEqual(configs[0].variant, gd.GradientDescentVariant.GD)
        self.assertEqual(configs[0].epochs, 50)
        self.assertEqual(configs[0].learning_rate, 0.3)
        self.assertEqual(configs[0].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[0].classes, [0, 9])

        self.assertEqual(configs[1].variant, gd.GradientDescentVariant.RGD)
        self.assertEqual(configs[1].epochs, 200)
        self.assertEqual(configs[1].learning_rate, 0.7)
        self.assertEqual(configs[1].coefficient, 0.5)
        self.assertEqual(configs[1].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[1].classes, [0, 9])

    def test_example_no_4(self):
        # Setting the command-line
        sys.argv = ['main.py',
                    '-v', 'SGD', '--epochs', '50', '--rate', '0.3', '--loss', 'hinge', '--digits', '8', '0',
                    '-v', 'PGD', '--epochs', '100', '--rate', '0.1', '--loss', 'bce', '--radius', '2', '--digits', '1', '2',
                    '-v', 'RGD', '--epochs', '200', '--rate', '0.05', '--loss', 'hinge', '--coefficient', '7.5', '--', 'digits', '7', '3']

        configs = main_utils.parse_arguments()

        self.assertEqual(len(configs), 3)

        self.assertEqual(configs[0].variant, gd.GradientDescentVariant.SGD)
        self.assertEqual(configs[0].epochs, 50)
        self.assertEqual(configs[0].learning_rate, 0.3)
        self.assertEqual(configs[0].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[0].classes, [8, 0])

        self.assertEqual(configs[1].variant, gd.GradientDescentVariant.PGD)
        self.assertEqual(configs[1].epochs, 100)
        self.assertEqual(configs[1].learning_rate, 0.1)
        self.assertEqual(configs[1].radius, 2)
        self.assertEqual(configs[1].loss_function, gd_utils.bce_loss)
        self.assertEqual(configs[1].classes, [1, 2])

        self.assertEqual(configs[2].variant, gd.GradientDescentVariant.RGD)
        self.assertEqual(configs[2].epochs, 200)
        self.assertEqual(configs[2].learning_rate, 0.05)
        self.assertEqual(configs[2].coefficient, 7.5)
        self.assertEqual(configs[2].loss_function, gd_utils.hinge_loss)
        self.assertEqual(configs[2].classes, [7, 3])



# class TestCountValidation(unittest.TestCase):
#     def test_shapes(self):
#         result main_utils._verify_variant_count_matches_paramerter_count(1, 1)
#         self.assertEqual(grads.shape, (NUM_WEIGHTS,))


if __name__ == '__main__':
    unittest.main()