import numpy as np
import pandas as pd
import warnings
import itertools

from ..association_rules import _tree

warnings.simplefilter(action='ignore', category=FutureWarning)

def cal_supports(freq, n_rows):
    """
    calculate supports

    Parameters
    ----------
    freq : int
        frequate of itemset.
    n_rows : int
        number of rows.

    Returns
    -------
    supports of itemset.

    """
    
    if n_rows == 0:
        raise ValueError("The rows supposed not to be zero")
    
    return freq/n_rows

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

def rearange_itemset(df):
    """
    rearange itemset with sum

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to be concatenated.

    Returns
    -------
    Dataframe with rearange columns.

    """
        
    columns = df.columns
    n_rows = df.shape[0]
        
    unique_supports = []
    for i, itemset in enumerate(columns):
        position_col = np.array(columns[i])
        comb_array = np.array(df.loc[:, position_col])
            
        frequent_itemset = np.sum(comb_array)
        support = cal_supports(frequent_itemset, n_rows)
            
        unique_supports.append(support)
            
    df = pd.concat([df, pd.DataFrame([unique_supports], columns=columns)])
    df = df.reset_index(drop=True)
    df = df.sort_values(n_rows, axis=1, ascending=False)
    df = df.drop(index=n_rows)
        
    return df

def compressing_itemset_tree(root, endswith):
    """
    compressing itemset with tree

    Parameters
    ----------
    root : tree.Node
        tree that has been fulled by itemset.
    endswith : str
        compress with endswith in column

    Returns
    -------
    dictionary of itemset and counts.

    """
        
    return root.find_set(endswith)

def make_tree(selected_df):
    """
    compressing itemset with tree

    Parameters
    ----------
    selected_df : pandas.DataFrame
        dataframe that has been selected.

    Returns
    -------
    root for tree.

    """
        
    root = _tree.Node(None)
        
    for i in selected_df.index:
        compressed = list(itertools.compress(selected_df.loc[i].index, selected_df.loc[i]))
            
        node = _tree.make_tree(compressed)
            
        root.check_add_child(node)
            
    return root

def selection_support_unique(compressed, unique, min_support):
    """
    selection support unique in compressed

    Parameters
    ----------
    compressed : dict
        compressed itemset and counts.
    unique : list
        unique itemset in compressed itemset.
    min_support : float
        minimal support to be selected

    Returns
    -------
    dict of selection support unique in compressed itemset.

    """
        
    unique_counts = {}
    length_compressed_keys = len(compressed.keys())
    for uniq in unique:
        unique_counts[uniq] = 0
        for key in compressed.keys():
            check_unique = any(uniq == np.array(key))
            if check_unique:
                unique_counts[uniq] += compressed[key]
                
        if cal_supports(unique_counts[uniq], length_compressed_keys) < min_support:
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
    endswith : object
        end of itemset.
    max_length : int
        max length of combinations

    Returns
    -------
    combinations.

    """
        
    if max_length == 0 or max_length == None:
        max_length = len(unique_counts.keys())
    
    keys_unique_counts = unique_counts.keys()
    combinations = []
    for i in range(1, max_length):
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
    combinations : list
        combinations of df columns.
    min_support : float
        minimal support to be select combinations

    Returns
    -------
    combinations and supports.

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
                
        support = cal_supports(length_check_array, n_rows)
                
        if support >= min_support:  
            selected_combinations.append(combination)
            selected_supports.append(support)
                
    return selected_combinations, selected_supports

def fpgrowth(df, min_support=0.5, return_df=False, max_length=5):
    """
    fpgrowth algorithm

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to get into algorithm.
            
        The Values are enter for example:
                
              Apple Banana Mango   
            1   1      1     0
            2   0      1     1
            3   0      1     0
            4   1      0     1
            5   0      1     1
    
    min_support : float, any float greater than 0
        minimal support to selection itemsets. The default is 0.5.
    return_df : bool, True or False
        returning as pandas dataframe
    max_length : int, 0 or None for auto length
        max length for combinations

    Returns
    -------
    supports of fpgrowth algorithm.

    """

    combinations_set = []
    supports = []
        
    df_copy = df.copy()
    columns = df_copy.columns
        
    df_copy = rearange_itemset(df_copy)
            
    root = make_tree(df_copy)
    for unique in columns:
        compressed_unique = compressing_itemset_tree(root, unique)
        unique_keys = [uniq for key in compressed_unique.keys() for uniq in key]
        selected_unique = selection_support_unique(compressed_unique, unique_keys, min_support)
        combinations = combinations_unique(selected_unique, unique, max_length)
                
        selected_combinations, selected_supports = selection_support_df(df_copy, combinations, min_support)
    
        combinations_set.extend(selected_combinations)
        supports.extend(selected_supports)
        
    combinations_supports = np.array([combinations_set, supports], dtype='object').T
    
    if return_df:
        col_output_df = ["Combinations", "Supports"]
        
        output_df = pd.DataFrame(combinations_supports, columns=col_output_df)
        
        return output_df
                
    return combinations_supports





