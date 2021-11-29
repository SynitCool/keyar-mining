import unittest
import numpy as np

from krmining.classification import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Input testing
        self.X = np.array(
            [
                [-0.94811012],
                [-0.62510428],
                [0.541039],
                [0.55501738],
                [-1.69834878],
                [-1.62414782],
                [-0.09021422],
                [0.4195677],
                [0.3861758],
                [1.78107583],
            ],
            dtype="float32",
        )

        self.y = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1], dtype="int32")

        # Expected output
        self.expected_predict = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

    def predict(self):
        # training the model
        model = LogisticRegression()
        model.fit(self.X, self.y)

        # predict the model
        y_pred = model.predict(self.X)

        # testing
        self.assertEqual(y_pred, self.expected_predict)
