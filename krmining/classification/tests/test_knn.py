import unittest

from krmining.datasets._dummy_data import make_dummy_data_classification
from krmining.classification._knn import KNearestNeighborsClassifier


class TestKNN(unittest.TestCase):
    def setUp(self):
        self.df = make_dummy_data_classification()

        self.k = 2

        self.expected_y = [1, 0]
        self.X_train = self.df.loc[[0, 1, 3, 4, 5, 6, 7, 9], [0]]
        self.y_train = self.df.loc[[0, 1, 3, 4, 5, 6, 7, 9], 1]

        self.X_test = self.df.loc[[2, 8], [0]]
        self.y_test = self.df.loc[[2, 8], 1]

    def test_knn(self):
        model = KNearestNeighborsClassifier(self.k)
        model.fit(self.X_train, self.y_train)

        model_pred = model.predict(self.X_test)

        self.assertEqual(self.expected_y, model_pred)
