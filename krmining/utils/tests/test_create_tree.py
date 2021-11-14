### IMPORTING CREATE TREE FUNCTIONS ###
from krmining.utils.create_tree import compressing_itemset_tree
from krmining.utils.create_tree import make_tree

from krmining.utils.tree import Node
from krmining.utils.association_rules_gen import rearange_itemset

### IMPORTING GENERAL PACKAGES ###
import unittest
import pandas as pd

# Test Class
class TestCreateTree(unittest.TestCase):
    def setUp(self):
        # Input testing
        self.df = pd.DataFrame(
            [[1, 0, 0], [1, 1, 1], [1, 0, 1]], columns=["apple", "banana", "mango"]
        )

        self.endswith = "banana"

        # Expected output
        self.expected_make_tree = ["apple", "mango", "banana"]
        self.expected_compressing_itemset_tree = {("mango",): 1}

    def test_make_tree(self):
        # Rearange df
        rearange_df = rearange_itemset(self.df)

        # Make tree
        root = make_tree(rearange_df)

        # Check if object type
        self.assertIsInstance(root, object)

        # Getting nodes
        result_nodes = root.get_nodes()

        # Check if is expected
        self.assertEqual(result_nodes, self.expected_make_tree)

    def test_compressing_itemset_tree(self):
        # Rearange df
        rearange_df = rearange_itemset(self.df)

        # Make tree
        root = make_tree(rearange_df)

        # Check if object type
        self.assertIsInstance(root, object)

        # Compressing itemset
        result_compressed = compressing_itemset_tree(root, self.endswith)

        # Check if as expected
        self.assertEqual(result_compressed, self.expected_compressing_itemset_tree)
