import unittest
import numpy as np
import gd_utils


NUM_SAMPLES = 100
NUM_WEIGHTS = 785

X = np.random.rand(NUM_SAMPLES, NUM_WEIGHTS)
Y_PRED = np.random.rand(NUM_SAMPLES)
Y_TRUE = np.random.rand(NUM_SAMPLES)


class TestBceGrad(unittest.TestCase):
    def test_shapes(self):
        grads = gd_utils.bce_grad(Y_PRED, Y_TRUE, X)
        self.assertEqual(grads.shape, (NUM_WEIGHTS,))  # add assertion here


class TestHingeGrad(unittest.TestCase):
    def test_shapes(self):
        grads = gd_utils.hinge_grad(Y_PRED, Y_TRUE, X)
        self.assertEqual(grads.shape, (NUM_WEIGHTS,))


if __name__ == '__main__':
    unittest.main()
