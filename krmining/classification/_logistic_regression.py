import numpy as np
import warnings

from krmining.utils.activation_funtions import sigmoid_function


class LogisticRegression:
    """
    Logistic Regression
    **For 2 classes only**
    """

    def __init__(self, learning_rate=0.001, epochs=1000):
        """
        Initialize logistic regression
        **For 2 classes only**
        in the further update will provide multi classes

        Parameters
        ----------
        learning_rate: float
            learning rate or how fast the model learning to update the model

        epochs: int
            iteration to learning for model


        Returns
        -------
        Initialize config the model


        Credits
        -------
        Origin of code: https://www.youtube.com/watch?v=JDU3AzH3WKg
        The link of repo: https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/logistic_regression.py
        """

        self.lr = learning_rate
        self.epochs = epochs
        self.__weights = None
        self.__bias = None

        warnings.warn(
            "The model still in maintaining in slow or extended memory", UserWarning
        )

    def fit(self, X, y):
        """
        Fitting the model or training the model

        Parameters
        ----------
        X: numpy.ndarray or shape (n_samples, n_features)
            The data to be trained.

        y: numpy.ndarray or shape (n_samples, )
            the target data of X

        Returns
        -------
        self: type
            the class
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

        # init parameters
        self.set_weights = np.zeros(n_features)
        self.set_intercept = 0

        # gradient descent
        for _ in range(self.epochs):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.intercept
            # apply sigmoid function
            y_predicted = sigmoid_function(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.set_weights -= self.lr * dw
            self.set_intercept -= self.lr * db

        return self

    def predict(self, X):
        """
        Predicting the model

        Parameters
        ----------
        X: numpy.ndarray or shape (n_samples, n_features)
            the data to be predicted

        Returns
        -------
        y_predicted_cls: list or shape(n_samples, )
            the predicted of X

        """
        X = np.array(X)

        if len(X.shape) != 2:
            raise ValueError(
                "shape of X supposed to be (n_samples, n_features) or reshape (n_samples, 1)"
            )

        if self.weights == None or self.intercept == None:
            raise NotImplementedError(
                "The model has not been trained, training the model first"
            )

        linear_model = np.dot(X, self.weights) + self.intercept
        y_predicted = sigmoid_function(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    @property
    def weights(self):
        return self.__weights

    @property
    def intercept(self):
        return self.__bias

    @weights.setter
    def set_weights(self, new_weights):
        self.__weights = new_weights

    @intercept.setter
    def set_intercept(self, new_intercept):
        self.__bias = new_intercept
