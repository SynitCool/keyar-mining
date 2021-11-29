import unittest

import numpy as np

from krmining.utils.least_square import calc_weights
from krmining.utils.least_square import calc_intercept

# Test class
class TestLeastSquare(unittest.TestCase):
    def setUp(self):
        # Input testing
        self.X = np.array(
            [
                [0.42521773],
                [0.10538228],
                [-1.09857726],
                [-0.23224205],
                [-0.29271159],
                [0.71534995],
                [1.27215258],
                [0.21612236],
                [-0.4150023],
                [1.00852528],
            ],
            dtype="float32",
        )

        self.y = np.array(
            [
                38.9461915,
                8.41264202,
                -60.17317552,
                -28.87914974,
                -3.51220467,
                32.06035364,
                64.18618897,
                -1.96365653,
                -24.80110076,
                65.67408856,
            ],
            dtype="float32",
        )

        # Expected output
        self.expected_weights = np.array([55.6223944], dtype="float32")
        self.expected_intercept = -0.48424515

    def test_calc_weights(self):
        # calculating weights
        result_weights = calc_weights(self.X, self.y)

        # testing
        np.testing.assert_almost_equal(result_weights, self.expected_weights, 5)

    def test_calc_intercept(self):
        # calculating intercept
        weights = calc_weights(self.X, self.y)
        result_intercept = calc_intercept(self.X, self.y, weights)

        # testing
        self.assertAlmostEqual(result_intercept, self.expected_intercept, 1)
