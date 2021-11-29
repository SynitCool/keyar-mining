import unittest

import numpy as np

from krmining.utils.metrics import sum_square_error


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Input testing
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
        self.pred_y = np.array(
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
            ],
            dtype="float32",
        )

        # expected output
        self.expected_sum_square_error = 1049.98169930904

    def test_sum_square_error(self):
        # calculation sum_square_error
        sse = sum_square_error(self.y, self.pred_y)

        # testing
        self.assertAlmostEqual(sse, self.expected_sum_square_error, 1)
