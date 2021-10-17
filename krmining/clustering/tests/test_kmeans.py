import numpy as np

import unittest

from krmining.clustering import KMeans
from krmining.datasets import get_example_covid_id


class TestKMeans(unittest.TestCase):
    def setUp(self):
        self.df = get_example_covid_id()

        self.k = 2

    def test_kmeans_notfitted(self):
        kmeans = KMeans(self.k)

        with self.assertRaises(AttributeError):
            centroids = kmeans.centroid

    def test_kmeans_random(self):
        kmeans = KMeans(self.k, init="random")
        kmeans.fit(self.df)
        centroid = kmeans.centroid

        expected_centroid = [
            [256449.25, 7341.5],
            [21714.566666666666, 541.2333333333333],
        ]

        centroid = np.ravel(centroid)
        expected_centroid = np.ravel(expected_centroid)

        self.assertAlmostEqual(sorted(list(centroid)), sorted(list(expected_centroid)))

    def test_kmeans_kmeans_plusplus(self):
        kmeans = KMeans(self.k, init="k-means++")
        kmeans.fit(self.df)
        centroid = kmeans.centroid

        expected_centroid = [[30757.5, 1099.34375], [346497.0, 5212.0]]

        self.assertAlmostEqual(sorted(list(centroid)), sorted(list(expected_centroid)))
