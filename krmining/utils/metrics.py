import numpy as np

import warnings


def sum_square_error(y, pred_y):
    """
    Sum Square Error

    Parameters
    ----------
    y: numpy.ndarray or shape (n_samples, )
        the actual y

    pred_y: numpy.ndarray or shape (n_samples, )
        the predicted from y

    Returns
    -------
    output: int
        the value of sse
    """

    y = np.array(y)
    pred_y = np.array(pred_y)

    if y.shape != pred_y.shape:
        raise ValueError("shape of y and predict y supposed to be same")

    error = y - pred_y
    square_error = error ** 2

    output = np.sum(square_error)

    return output
