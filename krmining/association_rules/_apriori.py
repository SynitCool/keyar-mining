import numpy as np
import pandas as pd
import warnings
import itertools

warnings.simplefilter(action="ignore", category=FutureWarning)


def cal_support(freq, n_rows):
    """
    Calculate support

    Parameters
    ----------
    freq : int
        frequation of data.
    n_rows : int
        rows of data.

    Returns
    -------
    support.

    """

    if n_rows == 0:
        raise ValueError("The rows supposed not to be zero")

    return freq / n_rows


def get_combinations(unique, length):
    """
    Get Combinations from unique data

    Parameters
    ----------
    unique : list or numpy.darray
        unique of data.
    length : int
        length of combinations.

    Returns
    -------
    Combinations with itemset length.

    """

    return itertools.combinations(unique, length)


def position_itemset(itemset, unique):
    """
    find position of itemset

    Parameters
    ----------
    itemset : tuple
        itemset of itemset.
    unique : list
        unique of itemset

    Returns
    -------
    position of itemset in unique.

    """
    positions = []
    for item in itemset:
        positions.append(np.where(item == unique)[0])

    return np.ravel(positions)


def apriori(df, max_length=5, min_support=0.5, return_df=False):
    """
    apriori algorithm

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to get into algorithm.
    max_length : int, any int greater than 0
        max length for combinations. The default is 5.
    min_support : float, any float greater than 0
        min support to selection. The default is 0.5.
    return_df : bool, False or True
        return as pandas dataframe. The default is False

    Returns
    -------
    supports of apriori algorithm.

    """

    warnings.warn(
        "The model still in maintaining in slow or extended memory", UserWarning
    )

    df = pd.DataFrame(df)

    combinations = []
    supports = []

    columns = df.columns
    n_rows = df.shape[0]

    for i in range(1, max_length + 1):
        gen_combinations = get_combinations(columns, i)
        for comb in gen_combinations:
            position = position_itemset(comb, columns)
            position_col = np.array(columns[position])
            length_col = len(position_col)
            comb_array = np.array(df.loc[:, position_col])

            check_array = np.where(length_col == comb_array.sum(axis=1))[0]
            length_check_array = len(check_array)

            support = cal_support(length_check_array, n_rows)

            # Selection support
            if support > min_support:
                combinations.append(comb)
                supports.append(support)

    combinations_supports = np.array([combinations, supports], dtype="object").T

    if return_df:
        col_output_df = ["Combinations", "Supports"]

        output_df = pd.DataFrame(combinations_supports, columns=col_output_df)

        return output_df

    return combinations_supports
