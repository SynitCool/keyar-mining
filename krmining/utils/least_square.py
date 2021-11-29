import numpy as np


def calc_weights(x, y):
    """
    Calculating weights in least square

    Parameters
    ----------
    x: numpy.ndarray or shape (n_samples, n_features)
        the data to be finding the weights for features.

    y: numpy.ndarray or shape (n_samples, )
        the target data of x.

    Returns
    -------
    output: numpy.ndarray or shape (n_features, )
        the weights of x
    """

    x = np.array(x)
    y = np.array(y)

    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)

    distance_x = x - mean_x
    distance_y = y - mean_y

    distance_xy = distance_x * distance_y.reshape((distance_y.shape[0], 1))

    x_square = (x - mean_x) ** 2

    numerator = np.sum(distance_xy, axis=0)
    denumerator = np.sum(x_square, axis=0)

    output = numerator / denumerator

    return output


def calc_intercept(X, y, slope):
    """
    Calculating intercept in least square

    Parameters
    ----------
    x: numpy.ndarray or shape (n_samples, n_features)
        the data to be finding the weights for features.

    y: numpy.ndarray or shape (n_samples, )
        the target data of x.

    slope: numpy.ndarray or shape (n_features, )
        the weighted of x


    Returns
    -------
    output: int
        the result of calculating intercept in least square

    """

    X = np.array(X)
    y = np.array(y)

    mean_y = y.mean(axis=0)
    mean_x = X.mean(axis=0)

    weighted_mean_x = slope * mean_x

    summing_weighted_mean_x = np.sum(weighted_mean_x)

    output = mean_y - summing_weighted_mean_x

    return output
