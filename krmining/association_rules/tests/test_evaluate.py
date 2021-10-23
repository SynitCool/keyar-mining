import unittest

from krmining.association_rules._fpgrowth import fpgrowth
from krmining.association_rules._evaluate import evaluate
from krmining.association_rules._apriori import apriori

from krmining.datasets import make_fruit_sold_dummy_association_rules


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.df = make_fruit_sold_dummy_association_rules()

        self.max_length = 2
        self.desire_confidence = [
            1.0,
            1.0,
            1.0,
            1.0,
            0.6666666666666666,
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            1.0,
            0.6666666666666666,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

    def test_evaluate_apriori(self):
        combinations_supports = apriori(self.df, max_length=self.max_length)
        apriori_evaluate = evaluate(combinations_supports, return_df=True)

        confidence = list(apriori_evaluate["confidence"])
        confidence = [round(confi, 2) for confi in confidence]
        confidence = sorted(confidence)

        desire_confidence = [round(confi, 2) for confi in self.desire_confidence]
        desire_confidence = sorted(desire_confidence)

        self.assertEqual(confidence, desire_confidence)

    def test_evaluate_fpgrowth(self):
        combinations_supports = fpgrowth(self.df, max_length=self.max_length)
        fpgrowth_evaluate = evaluate(combinations_supports, return_df=True)

        confidence = list(fpgrowth_evaluate["confidence"])
        confidence = [round(confi, 2) for confi in confidence]
        confidence = sorted(confidence)

        desire_confidence = [round(confi, 2) for confi in self.desire_confidence]
        desire_confidence = sorted(desire_confidence)

        self.assertEqual(confidence, desire_confidence)
