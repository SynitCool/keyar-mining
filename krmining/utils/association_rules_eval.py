import itertools
import warnings


def cal_support(freq, n_rows):
    """
    Calculate support

    Parameters
    ----------
    freq : int
        frequation of data.
            for example :
                32
    n_rows : int
        rows of data.
            for example :
                100

    Returns
    -------
    support.
        for example :
            = 32 / 100
            = 0.32


    """

    if n_rows == 0:
        raise ValueError("The rows supposed not to be zero")

    support = freq / n_rows

    return round(support, 3)


def make_combinations(combinations):
    """
    make combinations of combinations list

    Parameters
    ----------
    combinations : list
        combinations to make new combinations.
        for example :
            [("apple"), ("banana"), ("mango"),
            ("apple", "banana"), ...]

    Returns
    -------
    generator of combination.
    for example :
        antecedent_combination : [("apple"), ("banana"), ("mango")]
        consequent_combination : [
            [("apple"), ("banana"), ("mango"), ("apple", "banana"), ...]
            [("apple"), ("banana"), ("mango"), ("apple", "banana"), ...],
        ]
        combination : [
            [("apple"), ("apple", "banana"), ("apple", "mango"), ("apple", "banana"), ...]
            [("apple", "banana"), ("banana"), ("banana", "mango"), ("apple", "banana"), ...]
        ]

    """

    for first_combination in combinations:
        for second_combination in combinations:
            combination = set(first_combination)
            second_combination = set(second_combination)

            combination = combination.union(second_combination)

            combination = sorted(combination)
            antecedent_combination = sorted(first_combination)
            consequent_combination = sorted(second_combination)

            combination = tuple(combination)
            antecedent_combination = tuple(antecedent_combination)
            consequent_combination = tuple(consequent_combination)

            yield antecedent_combination, consequent_combination, combination


def find_supports(combinations, supports, combination):
    """
    find supports of combinations

    Parameters
    ----------
    combinations : list
        combinations to find supports.
            for example :
                [("apple"), ("banana"), ("mango"),
                 ("apple", "banana"), ...]

    supports : list
        support of combinations.
            for example :
                [0.43, 0.64, 0.35,
                 0.2, ...]

    combination : list
        combination to find support from combinations.
            for example :
                ("mango")

    Returns
    -------
    supports of combination.
        for example :
            = 0.35

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
            for example :
                - 0.43

    combination_support : float
        support of combination.
            for example :
                - 0.35

    Returns
    -------
    confidence of antecedents and combination.
        for example :
            = 0.35 / 0.43
            = 0.813

    """

    try:
        confidence = combination_support / antecedents_support

        return round(confidence, 3)
    except ZeroDivisionError:
        raise ValueError("antecedents support supposed not be zero !")
