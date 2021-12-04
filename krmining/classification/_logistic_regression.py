import numpy as np
import warnings

from krmining.utils.activation_funtions import sigmoid_function
from krmining.utils.optimizer import gradient_descent_optimizer


class LogisticRegression:
    """
    Logistic Regression
    **Try multiple class logistic regression**
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
        """

        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.intercept = []

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

        labels = np.unique(y)
        for label in labels:
            # finding index for current class as 1
            index_class = np.where(label == y)[0]
            X_1 = X[index_class]
            y_1 = np.ones((X_1.shape[0]), dtype="int32")

            # finding other for other class index as 0
            index_class = np.where(label != y)[0]
            X_0 = X[index_class]
            y_0 = np.zeros((X_0.shape[0]), dtype="int32")

            # concatenate X's and y's
            X_concat = np.concatenate([X_0, X_1])
            y_concat = np.concatenate([y_0, y_1])

            # gradient descent
            weights, intercept = gradient_descent_optimizer(
                X_concat, y_concat, self.lr, self.epochs
            )

            # appending
            self.weights.append(weights)
            self.intercept.append(intercept)

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
        probability: numpy.ndarray or shape(n_samples, n_labels)
            the probability of X

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

        # predicting using every weights and intercept
        probability = []
        for weight, bias in zip(self.weights, self.intercept):
            linear_model = np.dot(X, weight) + bias
            y_predicted = sigmoid_function(linear_model)

            probability.append(np.array(y_predicted))

        probability = np.array(probability)
        probability = probability.T

        return probability
