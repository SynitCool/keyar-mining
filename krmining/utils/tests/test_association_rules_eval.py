### IMPORTING FUNCTIONS OF ASSOCIATION RULES EVAL FUNCTIONS ###
from krmining.utils.association_rules_eval import cal_support
from krmining.utils.association_rules_eval import cal_confidence
from krmining.utils.association_rules_eval import make_combinations
from krmining.utils.association_rules_eval import find_supports


### IMPORTING GENERAL PACKAGE ###
import unittest
import types

# Test Class
class TestAssociationRulesEval(unittest.TestCase):
    def setUp(self):
        # Initialize input functions test
        self.n_rows = [0, 1000, 2000, 4000, 5000]
        self.freqs = [0, 25, 75, 45, 65]
        self.combinations = [
            [("apple",), ("banana",), ("mango",), ("apple", "banana")],
            [("",), ("", "jackfruit"), ("mango",), ("apple",)],
        ]
        self.supports = [[0.6, 0.7, 0.3, 0.6], [0.7, 0.8, 0.3, 0.65]]
        self.combination = [
            [("mango",), ("apple",), ("banana", "apple")],
            [("jackfruit",), ("mango",), ("apple",)],
        ]
        self.antecedents_supports = [[0.3, 0.75, 0.84, 0.43], [0, 0, 0.54, 0.2]]

        # Initialize expected outputs
        self.expected_cal_support = [ZeroDivisionError, 0.025, 0.037, 0.011, 0.013]
        self.expected_cal_confidence = [
            [2.0, 0.933, 0.357, 1.395],
            [ZeroDivisionError, ZeroDivisionError, 0.556, 3.25],
        ]
        self.expected_make_combinations = [
            [("apple",), ("apple", "banana"), ("apple", "mango"), ("apple", "banana")],
            [("",), ("", "jackfruit"), ("", "mango"), ("", "apple")],
        ]
        self.expected_find_supports = [[0.3, 0.6, 0.6], [0, 0.3, 0.65]]

    def test_cal_support(self):
        for freq, row, expected in zip(
            self.freqs, self.n_rows, self.expected_cal_support
        ):
            if row == 0:
                with self.assertRaises(ValueError):
                    cal_support(freq, row)

                continue

            support = cal_support(freq, row)
            self.assertEqual(support, expected)

    def test_cal_confidence(self):
        for i in range(len(self.supports)):
            for comb_support, ante_support, expected in zip(
                self.supports[i],
                self.antecedents_supports[i],
                self.expected_cal_confidence[i],
            ):
                if ante_support == 0:
                    with self.assertRaises(ValueError):
                        cal_confidence(ante_support, comb_support)

                    continue

                confidence = cal_confidence(ante_support, comb_support)
                self.assertEqual(confidence, expected)

    def test_make_combinations(self):
        for i in range(len(self.combinations)):
            generator = make_combinations(self.combinations[i])

            self.assertIsInstance(generator, types.GeneratorType)

            result_generator = [comb for ante, conse, comb in generator][:4]
            expected_generator = self.expected_make_combinations[i]

            self.assertEqual(result_generator, expected_generator)

    def test_find_supports(self):
        for i in range(len(self.combinations)):
            for comb, expected in zip(
                self.combination[i], self.expected_find_supports[i]
            ):
                found_support = find_supports(
                    self.combinations[i], self.supports[i], comb
                )

                self.assertEqual(found_support, expected)
