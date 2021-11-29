import numpy as np


def linear_function(data_x, weight, bias):
    """
    Linear function

    Y = mx + b

    Parameters
    ----------
    data_x: numpy.ndarray or shape (n_samples, n_features)
        the independent variables

    weight: numpy.ndarray or shape (n_features)
        the weight of the independent variables

    bias: int
        the intercept of dependent variable

    Returns
    -------
    output: numpy.ndarray or shape (n_samples, )
        the result of activation funtions
    """

    data_x = np.array(data_x)

    weighted_x = weight * data_x
    sum_weighted_x = np.sum(weighted_x, axis=1)

    output = bias + sum_weighted_x

    return output


def sigmoid_function(data_x):
    """
    Sigmoid Function

    1/(1 + exp(-x))

    Parameters
    ----------
    data_x: numpy.ndarray or shape (n_samples, n_features)
        the independent variables

    Returns
    -------
    output: numpy.ndarray or shape (n_samples, )
        result of sigmoid function

    """

    exp_num = np.exp(-data_x)
    denumerator = 1 + exp_num

    numerator = 1
    output = numerator / denumerator

    return output
