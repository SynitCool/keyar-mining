import numpy as np
import warnings


class KNearestNeighborsClassifier:
    """
    KNearestNeighborsClassifier Algorithm
    """

    def __init__(self, k):
        self.k = k

        self.status_trained = False

        warnings.warn(
            "The model still in maintaining in slow or extended memory", UserWarning
        )

    def fit(self, X, y):
        """
        Fitted the knn classifier

        Parameters
        ----------
        X : numpy.ndarray or shape (n_samples, n_features)
            training data.
        y : numpy.ndarray or shape (n_samples, )
            target data.

        Returns
        -------
        self : KNearestNeighborsClassifier
            Finish fitted KNearestNeighborsClassifier.

        """
        if len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError(
                "Shape of X should be (n_samples, n_features) and y should be (n_samples, )"
            )

        self.X_train = np.array(X)
        self.y_train = np.array(y)

        self.status_trained = True

        return self

    def predict(self, X):
        """
        predict use KNearestNeighborsClassifier

        Parameters
        ----------
        X : numpy.ndarray or shape (n_samples, n_features)
            predicted data.

        Raises
        ------
        AttributeError
            when the model is not fitted yet.

        Returns
        -------
        output_y : numpy.ndarray or shape (n_samples, )
            predicted from data using KNearestNeighborsClassifier.

        """

        if self.status_trained == False:
            raise AttributeError("The model is not fitted")
        elif len(X.shape) != 2:
            raise ValueError(
                "Shape of X should be (n_samples, n_features) and y should be (n_samples, )"
            )

        X = np.array(X)
        output_y = []
        for sample in X:
            # calculate sample and X_train
            distance_sample = self.__cal_euclidian_distance(sample, self.X_train)

            # zipped and sorted distance
            zipped_sample_label = np.dstack((distance_sample, self.y_train))
            zipped_sample_label = zipped_sample_label.reshape(
                (zipped_sample_label.shape[1], 2)
            )
            sorted_sample_label = sorted(zipped_sample_label, key=lambda x: x[0])[
                : self.k
            ]
            labels = np.array(sorted_sample_label)[:, 1].ravel()

            # counting labels
            labels, counts = np.unique(labels, return_counts=True)

            counts_labels = np.dstack((counts, labels))
            counts_labels = counts_labels.reshape((counts_labels.shape[1], 2))

            most_label_voted = self.__most_voted(counts_labels)

            output_y.append(most_label_voted)

        return output_y

    def evaluate(self, X, y):
        """
        evaluate model using KNearestNeighborsClassifier

        Parameters
        ----------
        X : numpy.ndarray or shape (n_samples, n_features)
            data to be evaluate.
        y : numpy.ndarray or shape (n_samples, )
            target data.

        Raises
        ------
        AttributeError
            when the model is not fitted.

        Returns
        -------
        evaluate_array : dict
            result evaluating model.

        """
        if self.status_trained == False:
            raise AttributeError("The model is not fitted")
        elif len(X.shape) != 2 or len(y.shape) != 1:
            raise ValueError(
                "Shape of X should be (n_samples, n_features) and y should be (n_samples, )"
            )

        y_predict = self.predict(X)

        confusion_matrix = {}
        for y_true, y_pred in zip(y, y_predict):
            try:
                if y_true == y_pred:
                    confusion_matrix[y_pred][0] += 1
                else:
                    confusion_matrix[y_pred][1] += 1
            except KeyError:
                confusion_matrix[y_pred] = [0, 0]

                if y_true == y_pred:
                    confusion_matrix[y_pred][0] += 1
                else:
                    confusion_matrix[y_pred][1] += 1

        print(confusion_matrix)

        all_tp = 0
        evaluate_array = {}
        for label in confusion_matrix:
            all_tp += confusion_matrix[label][0]

            precision = self.__cal_precision(
                confusion_matrix[label][0], confusion_matrix[label][1]
            )

            evaluate_array[label] = precision

        evaluate_array["accuracy"] = self.__cal_accuracy(all_tp, len(y))

        return evaluate_array

    def __cal_euclidian_distance(self, x, y):
        return list(map(np.linalg.norm, list(x - y)))

    def __most_voted(self, count_label):
        sorted_count_label = sorted(count_label, key=lambda x: x[0], reverse=True)

        most_voted_count_label = np.array(sorted_count_label)[0][1]

        return most_voted_count_label

    def __cal_precision(self, tp, fp):
        return tp / (tp + fp)

    def __cal_accuracy(self, tp, length_x):
        return tp / length_x
