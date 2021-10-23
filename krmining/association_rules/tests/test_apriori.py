import unittest

from krmining.association_rules._apriori import apriori

from krmining.datasets import make_fruit_sold_dummy_association_rules


class TestApriori(unittest.TestCase):
    def setUp(self):
        self.df = make_fruit_sold_dummy_association_rules()

        self.max_length = 2

    def test_apriori_itemset(self):
        supports = [
            round(support, 2)
            for support in apriori(self.df, max_length=self.max_length)[:, 1]
        ]

        desire_supports = [
            0.6666666666666666,
            1.0,
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            0.6666666666666666,
            1.0,
            0.6666666666666666,
            0.6666666666666666,
        ]

        desire_supports = [round(support, 2) for support in desire_supports]

        self.assertEqual(sorted(supports), sorted(desire_supports))
