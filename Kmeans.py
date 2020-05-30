from enum import Enum
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances, recall_score, f1_score
from collections import  defaultdict
import matplotlib.pyplot as plt

import numpy as np
from DataTidying import DataSet


class ALGO(Enum):
    DEFAULT = 0


class Kmeans:
    def __init__(self, maxiter=1000, distance="euclidean", seed=None):
        self.ALGO = ALGO.DEFAULT
        self.dataSource = DataSet()
        self.dataSource.load()
        self.random_seed = 0
        self.data = None
        self.k = None
        self.maxiter = maxiter
        self.record_heterogeneity = None
        self.clusters_radius = None
        self.verbose = False
        self.distance = distance
        self.seed = seed

    def init(self, factor):
        """
        explicitly resample data
        :param factor: factor of orginal data to resample
        """
        self.dataSource.sample(factor)
        self.data = self.dataSource.col_reduce_default()
        self.k = self.dataSource.k

    def kmeans_default(self):
        """
        Default implementation of kmeans, using sklearn
        :return:
        """
        X, y = self.data
        X = self.dataSource.normalize(X)
        k = self.k
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        kmeans = KMeans(n_clusters=k, random_state=0, init='random').fit(X_train)
        y_test = self.dataSource.remap(y_test, y_train, kmeans.labels_)
        y_test_pred = kmeans.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred, average='micro')
        f1 = f1_score(y_test, y_test_pred, average='micro')
        print("Categorizing into ", k, " clusters...")
        print("accuracy:", acc)
        print("recall:", rec)
        print("f1 score:", f1)

    def kmeans_mahalanobis(self, verbose=True):
        X, y = self.data
        X = self.dataSource.normalize(X)
        k = self.k
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        # init cluster
        self.distance = 'mahalanobis'
        self.maxiter = 300
        self.record_heterogeneity = []
        self.clusters_radius = []
        self.seed = 123
        self.verbose = verbose
        # fit models
        centroids, y_train_pred = self.fit(X_train, y_train)
        y_test = self.dataSource.remap(y_test, y_train, y_train_pred)
        y_test_pred = self.assign_clusters(X_test, centroids, self.distance)
        acc = accuracy_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=1)
        print("Categorizing into ", k, " clusters...")
        print("accuracy:", acc)
        print("recall:", rec)
        print("f1 score:", f1)

    def compute_heterogeneity(self, data, k, centroids, cluster_assignment, distance="euclidean"):
        """
        Function to computer heterogeneity of teh iterations of the KNN fit process
        :param data: all the points of the feature space to calcualte the heterogeneity
        :param k: number of clusters to compute
        :param centroids: position of baricenter of each cluster
        :param cluster_assignment: list of data points with their assignments within the clusters calculated in place.
        :param distance: euclidean or mahalabonious
        :return:
        """
        radius_list = []
        heterogeneity = 0.0
        for i in range(min(k, len(centroids))):
            # Select all data points that belong to cluster i. Fill in the blank
            member_data_points = data[cluster_assignment == i, :]
            if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
                # Compute distances from centroid to data points, based on the type of distance
                if distance == 'mahalanobis':
                    try:
                        vi = np.linalg.inv(np.cov(data.T)).T
                    except np.linalg.LinAlgError:
                        vi = np.linalg.pinv(np.cov(data.T)).T
                    distances = pairwise_distances(member_data_points, [centroids[i]], metric=distance, VI=vi)
                else:
                    distances = pairwise_distances(member_data_points, [centroids[i]], metric=distance)
                radius_list.append( np.max(np.array([abs(i) for i in distances])) )
                squared_distances = distances ** 2
                heterogeneity += np.sum(squared_distances)
        return heterogeneity, radius_list

    def revise_centroids(self, data, k, cluster_assignment):
        """
        After all points are assigned to a cluster, the centroids have to be revised to ensure that
            the distance between the points and the centroid is minimized.
        :param data: all the points of the feature space
        :param k: number of clusters to calculate
        :param cluster_assignment: list of cluster ids with the current assignment of the data points to the clusters
        :return:
        """
        new_centroids = []
        for i in range(k):
            # Compute the mean of the data points. Fill in the blank
            if len(data[cluster_assignment == i]) == 0:
                continue
            centroid = data[cluster_assignment == i].mean(axis=0)
            # Convert numpy.matrix type to numpy.ndarray type
            centroid = np.ravel(centroid)
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)
        return new_centroids

    def assign_clusters(self, data, centroids, distance):
        """
        Calculate the distance between each point to the centroids and decide to which cluster each point is assigned
        :param data: all the points of the feature space
        :param centroids: baricenter points of the different clusters
        :param distance: type of distance selected to be used to calculate the distance between the points and the centroids
        :return:
        """
        # Compute distances between each data point and the set of centroids, based on the distance selected:
        if distance == 'mahalanobis':
            try:
                vi = np.linalg.inv(np.cov(data.T)).T
            except np.linalg.LinAlgError:
                vi = np.linalg.pinv(np.cov(data.T)).T
            distances_from_centroids = pairwise_distances(data, centroids, metric=distance, VI=vi)
        else:
            distances_from_centroids = pairwise_distances(data, centroids, metric=distance)
        # Compute cluster assignments for each data point:
        cluster_assignment = np.argmin(distances_from_centroids, axis=1)
        return cluster_assignment

    def get_initial_centroids(self, data, k, labels, seed=None):
        """
        Randomly choose k data points as initial centroids
        :param data: all the points of the feature space
        :param labels: classfication label for stratify
        :param k: number of clusters to calculate
        :param seed: initial seed for the random number calculator
        :return:
        """
        if seed is not None:  # useful for obtaining consistent results
            np.random.seed(seed)
        # n = data.shape[0]  # number of data points
        # # Pick K indices from range [0, N).
        # rand_indices = np.random.randint(0, n, k)
        # # Keep centroids as dense format, as many entries will be nonzero due to averaging.
        # centroids = data[rand_indices, :]
        centroids = resample(data, n_samples=k, random_state=seed, replace=False, stratify=labels)
        return centroids

    def plot_all(self, mode="show"):
        """
        Plot all data.
        :return:
        """
        self.plot_radius(mode)
        self.plot_heterogeneity(mode)

    def plot_heterogeneity(self, mode="show"):
        """
        Function that allows to plot the evolution of the heterogeneity for each cluster wrt the number of iterations
        Inputs:
                - heterogeneity = List of heterogeneity values calculated during the fit of the model
                - k = number of clusters that have been calculated
        :return:
        """
        plt.figure(figsize=(7, 4))
        plt.plot(self.record_heterogeneity, linewidth=2)
        plt.xlabel('# Iterations')
        plt.ylabel('Heterogeneity')
        plt.title('Heterogeneity of clustering over time, K={0:d}'.format(self.k))
        plt.rcParams.update({'font.size': 12})
        plt.tight_layout()
        if mode == "show":
            plt.show()
        elif mode == "save":
            plt.savefig("heterogeneity_"+str(self.k))

    def plot_radius(self, mode="show"):
        """
        Function that allows to plot the radius for each cluster wrt the number of iterations
        Inputs:
                - heterogeneity = List of cluster radius values calculated during the fit of the model
                - k = number of clusters that have been calculated
        :return:
        """
        plt.figure(figsize=(7, 4))
        for r in zip(*self.clusters_radius):
            plt.plot(r, linewidth=2, c=np.random.rand(3,))
        plt.xlabel('# Iterations')
        plt.ylabel('Radius')
        plt.title('Radius of each cluster of clustering over time, K={0:d}'.format(self.k))
        plt.legend("upper right")
        plt.rcParams.update({'font.size': 12})
        plt.tight_layout()
        if mode == "show":
            plt.show()
        elif mode == "save":
            plt.savefig("radius_"+str(self.k))

    def fit(self, data, labels):
        """
        This function runs k-means on given data using a model of the class KNN.
        :param data:
        :param labels:
        :return:    - centroids = Cluster centroids that define the clusters that have been generated by the algorithm
                    - cluster assignments = List of points with their cluster id, defining which is the cluster they belong to.
        """
        centroids = self.get_initial_centroids(data=data, k=self.k, seed=self.seed, labels=labels)
        if self.verbose:
            print("Initial centroid number: ", len(centroids))
            print("Initial centroids: ", centroids)
        cluster_assignment = prev_cluster_assignment = None
        for itr in range(self.maxiter):
            if self.verbose:
                print("Iteration " + repr(itr) + ". Calculating the cluster assignments for all data points.")
            # 1. Make cluster assignments using nearest centroids
            cluster_assignment = self.assign_clusters(data=data, centroids=centroids, distance=self.distance)
            if self.verbose:
                dic= defaultdict(int)
                for l in cluster_assignment: dic[l] += 1
                print("Iteration ", itr, ". Size of each clusters:", dic)
            # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
            centroids = self.revise_centroids(data=data, k=self.k, cluster_assignment=cluster_assignment)
            # Check for convergence: if none of the assignments changed, stop
            if prev_cluster_assignment is not None and (prev_cluster_assignment == cluster_assignment).all():
                break
            # Print number of new assignments
            if prev_cluster_assignment is not None:
                num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
                if self.verbose:
                    print('    {0:5d} elements changed their cluster assignment during this assignment.'.format(
                        num_changed))
                    # Record heterogeneity convergence metric
            if self.record_heterogeneity is not None:
                score, radius_list = self.compute_heterogeneity(data=data, k=self.k, centroids=centroids,
                                                           cluster_assignment=cluster_assignment)
                self.record_heterogeneity.append(score)
                self.clusters_radius.append(radius_list)
            prev_cluster_assignment = cluster_assignment[:]
        return centroids, cluster_assignment

