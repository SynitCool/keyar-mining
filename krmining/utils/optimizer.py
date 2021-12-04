import numpy as np

from .activation_funtions import sigmoid_function


def gradient_descent_optimizer(X, y, lr, epochs):
    """
    Gradient Descent Optimizer
    **The optimizer is available for cross entropy only**

    Parameters
    ----------
    X: numpy.ndarray or shape (n_samples, n_features)
        The indepedent variables

    y: numpy.ndarray or shape (n_samples, n_features)
        The dependent variable

    lr: float
        the rate of optimizer to find the lowest point

    epochs: int
        The number of iteration until convergence

    Returns
    -------
    weights: numpy.ndarray or shape (n_features, )
        the weights of independent variables

    intercept: int
        the intercept of dependent variable

    """

    X = np.array(X)
    y = np.array(y)

    if len(X.shape) != 2:
        raise ValueError(
            "shape of X supposed to be (n_samples, n_features) or reshape (n_samples, 1)"
        )
    elif len(y.shape) != 1:
        raise ValueError("shape of y supposed to be (n_samples,)")

    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    intercept = 0

    # gradient descent
    for _ in range(epochs):
        # approximate y with linear combination of weights and x, plus intercept
        linear_model = np.dot(X, weights) + intercept
        # apply sigmoid function
        y_predicted = sigmoid_function(linear_model)

        # compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # update parameters
        weights -= lr * dw
        intercept -= lr * db

    return weights, intercept
