### IMPORTING TREE FUNCTIONS ###
from krmining.utils.tree import Node
from krmining.utils.tree import make_tree
from krmining.utils.association_rules_gen import rearange_itemset

### IMPORTING GENERAL FUNCTIONS ###
import unittest
import inspect
import pandas as pd

# Test Class
class TestTree(unittest.TestCase):
    def setUp(self):
        # Input testing
        self.df = pd.DataFrame(
            [[1, 0, 0], [1, 1, 1], [1, 0, 1]], columns=["apple", "banana", "mango"]
        )

        # Expected output
        self.expected_get_nodes = ["apple", "mango", "banana"]

    def test_get_nodes(self):
        rearange_df = rearange_itemset(self.df)

        root = make_tree(rearange_df)

        self.assertIsInstance(root, object)

        result_nodes = root.get_nodes()

        self.assertEqual(result_nodes, self.expected_get_nodes)
