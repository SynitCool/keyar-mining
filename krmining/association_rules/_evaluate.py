import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def make_combinations(combinations):
    """
    make combinations of combinations list

    Parameters
    ----------
    combinations : list
        combinations to make new combinations.

    Returns
    -------
    generator of combination.

    """
    
    for first_combination in combinations:
        for second_combination in combinations:
            combination = set(first_combination)
            second_combination = set(second_combination)
                
            combination = combination.union(second_combination)
                
            combination = tuple(combination)
            antecedent_combination = tuple(first_combination)
            consequent_combination = tuple(second_combination)
                
            if set(antecedent_combination) == set(consequent_combination):
                continue
                
            yield antecedent_combination, consequent_combination, combination

def find_supports(combinations, supports, combination):
    """
    find supports of combinations

    Parameters
    ----------
    combinations : list
        combinations to find supports.
    supports : list
        support of combinations.
    combination : list
        combination to find support from combinations

    Returns
    -------
    supports of combination.

    """
    for i, comb in enumerate(combinations):
        if set(comb) == set(combination):
            return supports[i]
                
    return 0

def cal_confidence(antecedents_support, combination_support):
    """
    calculate confidence of antecedents and consequents

    Parameters
    ----------
    antecedents_support : float
        support of antecedents.
    combination_support : float
        support of combination.

    Returns
    -------
    confidence of antecedents and combination.

    """
        
    try:
        return combination_support / antecedents_support
    except ZeroDivisionError:
        return 0

def evaluate(combinations_supports, min_threshold=0.5, return_df=False):
    """
    evaluating supports of association rules algorithm

    Parameters
    ----------
    combinations_supports : numpy.darray or pandas.DataFrame
        the combinations supports to be evaluate.
    min_threshold : float, any number greater than 0
        selection the metrics or evaluate from combinations. The default is 0.5.
    return_df : bool, True or False
        returning as pandas.DataFrame or not. The default is False.

    Returns
    -------
    returning metrices or combinations and supports.

    """
    combinations_supports = np.array(combinations_supports)
    
    combinations = combinations_supports[:, 0]
    supports = combinations_supports[:, 1]
    
    antecedents_combinations = []
    consequents_combinations = []
    combinations_set = []
        
    antecedents_supports = []
    consequents_supports = []
    combinations_supports =[]
        
    confidence = []
        
    generator_combinations = make_combinations(combinations)
    for antecedent, consequent, combination in generator_combinations:
        antecedent_support = find_supports(combinations, supports, antecedent)
        consequent_support = find_supports(combinations, supports, consequent)
        combination_support = find_supports(combinations, supports, combination)
        
        confidence_set = cal_confidence(antecedent_support, combination_support)
            
        if (antecedent_support > min_threshold and
            consequent_support > min_threshold and
            combination_support > min_threshold and 
            confidence_set > min_threshold):
            
            antecedents_combinations.append(antecedent)
            consequents_combinations.append(consequent)
            combinations_set.append(combination)
                    
            antecedents_supports.append(antecedent_support)
            consequents_supports.append(consequent_support)
            combinations_supports.append(combination_support)
                    
            confidence.append(confidence_set)
                
                
        
    tables = np.array([antecedents_combinations, 
                       consequents_combinations, 
                       combinations_set, 
                       antecedents_supports,
                       consequents_supports,
                       combinations_supports,
                       confidence], dtype='object').T
        
    columns = ["antecedents", "consequents", 
               "combination", "antecedent support", 
               "consequent support", "combination support",
               "confidence"]
        
    return pd.DataFrame(tables, columns=columns) if return_df else tables
    






















