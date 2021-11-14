import numpy as np
import pandas as pd

import itertools
import warnings

from .association_rules_eval import cal_support


def get_combinations(unique, length):
    """
    Get Combinations from unique data

    Parameters
    ----------
    unique : list or numpy.darray
        unique of data.
            for example :
                = ["apple", "banana", "mango"]
    length : int
        length of combinations.
            for example :
                = 2

    Returns
    -------
    Combinations with itemset length.
        for example :
            = [("apple", "banana"), ("apple", "mango"), ("banana", "mango")]

    """

    return itertools.combinations(unique, length)


def position_itemset(itemset, unique):
    """
    find position of itemset

    Parameters
    ----------
    itemset : tuple
        itemset of itemset.
            for example :
                = ("apple", "banana")

    unique : list
        unique of itemset
            for example :
                = ["apple", "mango", "banana", "strawberry"]

    Returns
    -------
    position of itemset in unique.
        for example :
            = [0, 2]

    """
    positions = []
    for item in itemset:
        positions.append(np.where(item == unique)[0])

    return np.ravel(positions)


def rearange_itemset(df):
    """
    rearange itemset with sum

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be concatenated.
            for example :
                | apple | banana |
                |   1   |    1   |
                |   0   |    1   |
                |   0   |    1   |


    Returns
    -------
    Dataframe with rearange columns.
        for example :
            | banana | apple |
            |    1   |   1   |
            |    1   |   0   |
            |    1   |   0   |

    """

    columns = df.columns
    n_rows = df.shape[0]

    unique_supports = []
    for i, itemset in enumerate(columns):
        position_col = np.array(columns[i])
        comb_array = np.array(df.loc[:, position_col])

        frequent_itemset = np.sum(comb_array)
        support = cal_support(frequent_itemset, n_rows)

        unique_supports.append(support)

    df = pd.concat([df, pd.DataFrame([unique_supports], columns=columns)])
    df = df.reset_index(drop=True)
    df = df.sort_values(n_rows, axis=1, ascending=False)
    df = df.drop(index=n_rows)

    return df


def selection_support_unique(compressed, unique, min_support):
    """
    selection support unique in compressed

    Parameters
    ----------
    compressed : dict
        compressed itemset and counts.
            for example :
                = {"apple": 2, "banana": 1, "mango": 3, "strawberry": 4}

    unique : list
        unique itemset in compressed itemset.
            for example :
                = ["apple", "banana", "mango", "strawberry"]

    min_support : float
        minimal support to be selected.
            for example :
                = 0.5

    Returns
    -------
    dict of selection support unique in compressed itemset.
        for example :
            = {"apple": 0.5, "mango": 0.75, "strawberry": 0.1}

    """

    unique_counts = {}
    length_compressed_keys = len(compressed.keys())
    for uniq in unique:
        unique_counts[uniq] = 0
        for key in compressed.keys():
            check_unique = any(uniq == np.array([key]))
            if check_unique:
                unique_counts[uniq] += compressed[key]

        if cal_support(unique_counts[uniq], length_compressed_keys) < min_support:
            unique_counts.pop(uniq)
        else:
            unique_counts[uniq] = unique_counts[uniq] / length_compressed_keys

    return unique_counts


def combinations_unique(unique_counts, endswith, max_length):
    """
    find combinations unique

    Parameters
    ----------
    unique_counts : unique_counts
        unique_counts to find combinations.
            for example :
                = {"apple": 0.66, "mango": 0.1}

    endswith : object
        end of itemset.
            for example :
                = "apple"

    max_length : int
        max length of combinations
            for example :
                = 2

    Returns
    -------
    combinations.
        for example :
            [("apple", "apple"), ("mango", "apple"),
            ("apple", "mango", "apple"), ("apple", )]

    """

    if max_length == 0 or max_length == None:
        max_length = len(unique_counts.keys())

    keys_unique_counts = unique_counts.keys()
    combinations = []
    for i in range(1, max_length + 1):
        combination = itertools.combinations(keys_unique_counts, i)
        for comb in combination:
            comb = list(comb)
            comb.append(endswith)
            comb = tuple(comb)

            combinations.append(comb)

    combinations.append((endswith,))

    return combinations


def selection_support_df(df, combinations, min_support):
    """
    selection combinations with support

    Parameters
    ----------
    df : pandas.DataFrame
        data to be selected.
            for example :
                = | banana |  mango | apple |
                  |   1    |    1   |   1   |
                  |   1    |    0   |   0   |
                  |   1    |    1   |   0   |

    combinations : list
        combinations of df columns.
            for example :
                = [("apple", "apple"), ("banana", "apple"), ("mango", "apple")
                   ("apple", "banana", "apple"), ("apple", "mango", "apple"),
                   ("banana", "mango", "apple"), ("apple",), ...]

    min_support : float
        minimal support to be select combinations
            for example :
                = 0.5

    Returns
    -------
    combinations and supports.
        for example :
            = [("banana", "mango", "apple"), ...]
            = [0.1, ...]

    """

    selected_supports = []
    selected_combinations = []

    columns = df.columns
    n_rows = df.shape[0]
    for combination in combinations:
        position = position_itemset(combination, columns)
        position_columns = np.array(columns[position])
        length_combination = len(combination)

        combination_array = np.array(df.loc[:, position_columns])

        check_array = np.where(length_combination == combination_array.sum(axis=1))[0]
        length_check_array = len(check_array)

        support = cal_support(length_check_array, n_rows)

        if support >= min_support:
            selected_combinations.append(combination)
            selected_supports.append(support)

    return selected_combinations, selected_supports
