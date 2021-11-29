import unittest

import numpy as np

from krmining.utils.activation_funtions import linear_function
from krmining.utils.activation_funtions import sigmoid_function


class TestActivationFunctions(unittest.TestCase):
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
        self.weights = np.array([55.6223944], dtype="float32")
        self.bias = -0.48424515

        # Expecting output
        self.expected_linear_function = np.array(
            [
                23.16738316,
                5.37736943,
                -61.58974268,
                -13.40210423,
                -16.76556477,
                39.30523211,
                70.27592733,
                11.53699788,
                -23.56766664,
                55.61234588,
            ]
        )

        self.expected_sigmoid_function = np.array(
            [
                [0.60473114],
                [0.52632123],
                [0.2500066],
                [0.44219905],
                [0.42734012],
                [0.6715822],
                [0.781111],
                [0.55382127],
                [0.39771327],
                [0.73273146],
            ]
        )

    def test_linear_function(self):
        # use linear function
        activated = linear_function(self.X, self.weights, self.bias)

        # testing
        np.testing.assert_almost_equal(activated, self.expected_linear_function, 5)

    def test_sigmoid_function(self):
        # use sigmoid function
        activated = sigmoid_function(self.X)

        # testing
        np.testing.assert_almost_equal(activated, self.expected_sigmoid_function)
