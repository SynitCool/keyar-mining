from ..utils import tree

import itertools


def compressing_itemset_tree(root, endswith):
    """
    compressing itemset with tree

    Parameters
    ----------
    root : tree.Node
        tree that has been fulled by itemset.
            for example : 
                apple
               //   \\
             banana
             //  \\
           mango

    endswith : str
        compress with endswith in column.
            for example : 
                = "mango"

    Returns
    -------
    dictionary of itemset and counts.
        for example :
            {"mango": 1}

    """

    return root.find_set(endswith)


def make_tree(selected_df):
    """
    compressing itemset with tree

    Parameters
    ----------
    selected_df : pandas.DataFrame
        dataframe that has been selected.
            for example : 
                | apple | banana |  mango |
                |   1   |    0   |    0   | 
                |   1   |    1   |    1   | 
                |   1   |    0   |    1   |
    Returns
    -------
    root for tree.
        for example : 
            apple
           //   \\
        banana
        //  \\
      mango
    """

    root = tree.Node(None)

    for i in selected_df.index:
        compressed = list(
            itertools.compress(selected_df.loc[i].index, selected_df.loc[i])
        )

        node = tree.make_tree(compressed)

        root.check_add_child(node)

    return root
