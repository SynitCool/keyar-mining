import numpy as np
import warnings


class KMeans:
    def __init__(self, k, n_iters=10, init="k-means++"):
        """
        Parameters
        ----------
        k : int
            number of clusters
        n_iters : int, int number greater than 0
            the iteration of k-means algorithm. The default is 10.
        init : string, {"random", "kmeans++"}
            the algorith to determine first centroids. The default is "kmeans++".
        """

        self.k = k
        self.n_iters = n_iters
        self.init = init

        warnings.warn(
            "The model still in maintaining in slow or extended memory", UserWarning
        )

    def fit(self, data):
        """
        This the kmeans algorithm

        Step 1 = select k
        Step 2 = pick random centroid
        Step 3 = calculate distance between data and centroid
        Step 4 = calculate the centroid
        Step 5 = Iterate from step 2 until the values is not changing

        """

        self.centroid = []

        X_copy = np.array(data)

        if self.init == "k-means++":
            centroids = self.__init_kmeans(data)
        else:
            random_indexes = np.random.choice(X_copy.shape[0], self.k, replace=False)

            centroids = [list(X_copy[index]) for index in random_indexes]

        for i in range(self.n_iters):
            old_centroids = centroids.copy()
            distance_centroid = []
            centroids_index = [[] for _ in range(self.k)]

            # Calculate all data to each centroid
            for centroid in centroids:
                distance = self.__euclidian_distance(X_copy, centroid)

                distance_centroid.append(distance)

            # Find nearest data to cluster
            clusters = np.where(
                np.array(distance_centroid).T
                == np.amin(np.array(distance_centroid).T, axis=1).reshape(-1, 1)
            )

            clusters = clusters[1]

            # update centroids
            for i, index in enumerate(centroids_index):
                indexes = np.where(clusters == i)[0]

                centroids_index[i] = list(indexes)

                data_index = X_copy[centroids_index[i]].mean(axis=0)

                centroids[i] = list(data_index)

            if old_centroids == centroids:
                break

        self.centroid = centroids

        return self

    def predict(self, data):
        X_copy = np.array(data)
        distance_centroid = []
        clusters = None

        for i, centroid in enumerate(self.centroid):
            distance = self.__euclidian_distance(X_copy, centroid)

            distance_centroid.append(distance)

        clusters = np.where(
            np.array(distance_centroid).T
            == np.amin(np.array(distance_centroid).T, axis=1).reshape(-1, 1)
        )

        indexes = clusters[0]
        clusters = clusters[1]

        list_unique = [np.where(index == indexes)[0] for index in np.unique(indexes)]

        list_of_dup = list(filter(lambda inds: len(inds) > 1, list_unique))

        dup_indexes = [i for l in list_of_dup for i in l[1:]]

        indexes = np.delete(indexes, dup_indexes)
        clusters = np.delete(clusters, dup_indexes)

        return clusters

    def evaluate(self, data):
        X_copy = np.array(data)
        predicted = self.predict(X_copy)
        clusters = predicted
        distance_centroid = []
        sse_clusters = []
        sse_all_clusters = None

        for i, centroid in enumerate(self.centroid):
            indexes = np.where(clusters == i)[0]

            distance = list(map(np.linalg.norm, list(X_copy[indexes] - centroid)))

            distance_centroid.append(distance)

        for distance in distance_centroid:
            sse_clusters.append(np.sum(np.square(distance)))

        sse_all_clusters = np.sum(sse_clusters)

        evaluate_info = {}

        for i, cluster_distance in enumerate(sse_clusters):
            evaluate_info[f"sse_cluster_{i}"] = sse_clusters[i]
            evaluate_info["sse_all_cluster"] = sse_all_clusters

        return evaluate_info

    def __init_kmeans(self, data):
        X_copy = np.array(data)

        centroids = []

        random_index = np.random.choice(X_copy.shape[0], replace=False)

        centroids.append(list(X_copy[random_index]))

        for i in range(self.k - 1):
            distances = []
            indexes_distances = []
            for centroid in centroids:
                distances_centroid = self.__euclidian_distance(X_copy, centroid)

                distances.append(distances_centroid)

            nearest_cluster = np.where(
                np.array(distances).T
                == np.amin(np.array(distances).T, axis=1).reshape(-1, 1)
            )

            indexes = nearest_cluster[0]
            clusters = nearest_cluster[1]
            unique_clusters = np.unique(clusters)

            dup_indexes = [
                np.where(indexes == index)[0] for index in np.unique(indexes)
            ]

            dup_indexes = list(filter(lambda inds: len(inds) > 1, dup_indexes))
            dup_indexes = [index for indexes in dup_indexes for index in indexes[1:]]

            indexes = np.delete(indexes, dup_indexes)
            clusters = np.delete(clusters, dup_indexes)

            for cluster in unique_clusters:
                indexes = np.where(clusters == cluster)[0]

                distances[cluster] = list(np.array(distances[cluster])[indexes])
                indexes_distances.append(indexes)

            distances = np.array([d for distance in distances for d in distance])
            indexes_distances = np.array(
                [d for distance in indexes_distances for d in distance]
            )

            distances = distances / np.sum(distances)

            max_index_distances = np.where(distances == distances.max())[0]
            max_index = indexes_distances[max_index_distances][0]

            centroids.append(list(X_copy[max_index]))

        return centroids

    def __euclidian_distance(self, x, y):
        return list(map(np.linalg.norm, list(x - y)))
