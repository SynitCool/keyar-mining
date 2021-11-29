import unittest
import numpy as np

from krmining.regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
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

        # Expected testing
        self.expected_predict = np.array(
            [
                23.167381,
                5.377369,
                -61.58975,
                -13.402106,
                -16.765568,
                39.305237,
                70.27593,
                11.536998,
                -23.567669,
                55.612347,
            ],
            dtype="float32",
        )
        self.expected_evaluate = {"SSE": np.float32(1049.9819)}
        self.expected_weights = np.array([55.6224], dtype="float32")
        self.expected_intercept = -0.48424625

    def test_predict(self):
        # fitting linear regression
        lr = LinearRegression()
        lr.fit(self.X, self.y)

        # making prediction
        pred_y = lr.predict(self.X)

        # testing
        np.testing.assert_almost_equal(pred_y, self.expected_predict)

    def test_evaluate(self):
        # fitting linear regression
        lr = LinearRegression()
        lr.fit(self.X, self.y)

        # making evaluate
        metrics = lr.evaluate(self.X, self.y)

        # testing
        self.assertDictEqual(metrics, self.expected_evaluate)

    def test_weights(self):
        # fitting linear regression
        lr = LinearRegression()
        lr.fit(self.X, self.y)

        # get weights
        result_weights = lr.weights

        # testing
        self.assertEqual(result_weights, self.expected_weights)

    def test_intercept(self):
        # fitting linear regression
        lr = LinearRegression()
        lr.fit(self.X, self.y)

        # get intercept
        result_intercept = lr.intercept

        # testing
        self.assertAlmostEqual(result_intercept, self.expected_intercept)
