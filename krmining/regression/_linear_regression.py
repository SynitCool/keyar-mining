import numpy as np
import warnings

from krmining.utils.activation_funtions import linear_function
from krmining.utils.least_square import calc_weights
from krmining.utils.least_square import calc_intercept
from krmining.utils.metrics import sum_square_error


class LinearRegression:
    """
    Linear Regression
    """

    def __init__(self):
        """
        Initialize config of linear regression

        Returns
        -------
        self: type
            the class
        """
        warnings.warn(
            "The model still in maintaining in slow or extended memory", UserWarning
        )

        self.__weights = []
        self.__intercept = None

    def fit(self, x, y):
        """
        Fitting linear regression

        Parameters
        ----------
        x: numpy.ndarray or shape (n_samples, n_features)
            the independent variables.

        y: numpy.ndarray or shape (n_samples, )
            the target data or dependent variabels of x.

        Returns
        -------
        self: type
            the fitting class

        """
        x = np.array(x)
        y = np.array(y)

        if len(x.shape) != 2:
            raise ValueError(
                "shape of x supposed to be (n_samples, n_features) or reshape (n_samples, 1)"
            )
        elif len(y.shape) != 1:
            raise ValueError("shape of y supposed to be (n_samples,)")

        weights = calc_weights(x, y)
        intercept = calc_intercept(x, y, weights)

        self.set_weights = weights
        self.set_intercept = intercept

        return self

    def predict(self, x):
        """
        Predict independent variables

        Parameters
        ----------
        x: numpy.ndarray or shape (n_samples, n_features)
            the independent variables to be predicted

        Returns
        -------
        predicted_y: numpy.ndarray in shape (n_samples, )
            the predicted of x
        """
        # declare var
        x = np.array(x)

        # check error
        if len(x.shape) != 2:
            raise ValueError(
                "shape of x supposed to be (n_samples, n_features) or reshape to (n_samples, 1)"
            )

        # check if model is trained
        if self.weights == [] or self.intercept == None:
            raise ValueError("the model should be trained first")

        # predict
        predicted_y = linear_function(x, self.weights, self.intercept)

        return predicted_y

    def evaluate(self, x, y):
        """
        Evaluate linear regression

        Parameters
        ----------
        x: numpy.ndarray or shape (n_samples, n_features)
            the independent variables.

        y: numpy.ndarray or shape (n_samples, )
            the target data or dependent variabels of x.

        """
        # declare var
        x = np.array(x)
        y = np.array(y)

        # checking error
        if len(x.shape) != 2:
            raise ValueError(
                "shape of x supposed to be (n_samples, n_features) or reshape (n_samples, 1)"
            )
        elif len(y.shape) != 1:
            raise ValueError("shape of y supposed to be (n_samples,)")

        # check if model is trained
        if self.weights == [] or self.intercept == None:
            raise ValueError("the model should be trained first")

        # predict
        predicted_y = self.predict(x)

        # calculating metrics
        sse = sum_square_error(y, predicted_y)

        # declare metrics var
        metrics = {"SSE": sse}

        return metrics

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def set_weights(self, new_weights):
        self.__weights = new_weights

    @property
    def intercept(self):
        return self.__intercept

    @intercept.setter
    def set_intercept(self, new_intercept):
        self.__intercept = new_intercept
