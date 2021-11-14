### IMPORTING ASSOCIATION RULES GEN FUNCTIONS ###
from krmining.utils.association_rules_gen import get_combinations
from krmining.utils.association_rules_gen import position_itemset
from krmining.utils.association_rules_gen import rearange_itemset
from krmining.utils.association_rules_gen import selection_support_unique
from krmining.utils.association_rules_gen import combinations_unique
from krmining.utils.association_rules_gen import selection_support_df

### IMPORTING GENERAL PACKAGE ###
import unittest
import types
import collections
import pandas as pd
import numpy as np


# Test Class
class TestAssociationRulesGen(unittest.TestCase):
    def setUp(self):
        # Input
        self.unique = [["apple", "banana", "mango"], ["", "jackfruit", "apple"]]
        self.itemset = [
            [("apple", "banana", "mango"), ("apple", "banana"), ("mango", "banana")],
            [("jackfruit", "apple"), ("apple",), ("jackfruit",)],
        ]
        self.df = pd.DataFrame(
            [[1, 1, 1], [1, 0, 0], [1, 1, 0]], columns=["banana", "mango", "apple"]
        )
        self.unique_counts = [
            {"apple": 2, "banana": 1, "mango": 3},
            {"": 1, "jackfruit": 2, "apple": 3},
        ]
        self.min_support = 0.5
        self.endswith = ["apple", "jackfruit"]
        self.max_length = 2
        self.combinations = [
            ("banana", "mango"),
            ("mango", "apple"),
            ("banana", "mango", "apple"),
        ]

        # Expected Output
        self.expected_get_combinations = [
            [("apple", "banana"), ("apple", "mango"), ("banana", "mango")],
            [("", "jackfruit"), ("", "apple"), ("jackfruit", "apple")],
        ]
        self.expected_position_itemset = [
            [[0, 1, 2], [0, 1], [2, 1]],
            [[1, 2], [2], [1]],
        ]
        self.expected_rearange_itemset = ["banana", "mango", "apple"]
        self.expected_selection_support_unique = [
            {"apple": 0.6666666666666666, "mango": 1.0},
            {"jackfruit": 0.6666666666666666, "apple": 1.0},
        ]
        self.expected_combinations_unique = [
            [
                ("apple", "apple"),
                ("mango", "apple"),
                ("apple", "mango", "apple"),
                ("apple",),
            ],
            [
                ("jackfruit", "jackfruit"),
                ("apple", "jackfruit"),
                ("jackfruit", "apple", "jackfruit"),
                ("jackfruit",),
            ],
        ]
        self.expected_selection_support_df = [[("banana", "mango")], [0.667]]

    def test_get_combinations(self):
        # Looping number testing
        for i in range(len(self.unique)):
            # Testing function
            generator = get_combinations(self.unique[i], self.max_length)

            # Check if type is iterator
            self.assertIsInstance(generator, collections.Iterator)

            # Result of generator
            result_generator = list(generator)

            # Check if result generator same as expected
            self.assertEqual(result_generator, self.expected_get_combinations[i])

    def test_position_itemset(self):
        # Looping number testing
        for i in range(len(self.itemset)):
            for item, expected in zip(
                self.itemset[i], self.expected_position_itemset[i]
            ):
                # Result position
                result_position = position_itemset(item, np.array(self.unique[i]))

                # Check if result position same as expected
                self.assertEqual(list(result_position), expected)

    def test_rearange_itemset(self):
        # Result dataframe rearange
        result_df = rearange_itemset(self.df)

        # Check if it is trully dataframe
        self.assertIsInstance(result_df, pd.DataFrame)

        # Result columns
        result_columns = result_df.columns

        # Check if result columns as expected
        self.assertEqual(list(result_columns), self.expected_rearange_itemset)

    def test_selection_support_unique(self):
        # Looping number testing
        for i in range(len(self.unique_counts)):
            # Result of selection
            result_selection_support = selection_support_unique(
                self.unique_counts[i], np.array(self.unique[i]), self.min_support
            )

            # Check if as expected
            self.assertEqual(
                result_selection_support, self.expected_selection_support_unique[i]
            )

    def test_combinations_unique(self):
        # Looping number testing
        for i in range(len(self.unique_counts)):
            # Selection unique counts
            selected_unique_counts = selection_support_unique(
                self.unique_counts[i], self.unique[i], self.min_support
            )

            # Combinations unique
            result_combinations = combinations_unique(
                selected_unique_counts, self.endswith[i], self.max_length
            )

            # Check if as expected
            self.assertEqual(result_combinations, self.expected_combinations_unique[i])

    def test_selection_support_df(self):
        # Unpack expected values
        expected_combinations, expected_supports = self.expected_selection_support_df

        # Selection
        result_combinations, result_supports = selection_support_df(
            self.df, self.combinations, self.min_support
        )

        # Check if as expected
        self.assertEqual(result_combinations, expected_combinations)
        self.assertEqual(result_supports, expected_supports)
