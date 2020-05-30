
from sklearn import datasets
from Kmeans import Kmeans
from configparser import ConfigParser

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score




if __name__ == '__main__':
    km = Kmeans()
    # km.init(0.0001)
    # km.kmeans_default()
    # km.init(0.001)
    # km.kmeans_default()
    # km.init(0.01)
    # km.kmeans_default()
    # km.init(0.1)
    # km.kmeans_default()
    # km.init(1)
    # km.kmeans_default()
    km.init(0.001)
    km.kmeans_mahalanobis()
    km.plot_all("save")
    km.init(0.01)
    km.kmeans_mahalanobis()
    km.plot_all("save")
    km.init(0.1)
    km.kmeans_mahalanobis()
    km.plot_all("save")
    km.init(1)
    km.kmeans_mahalanobis()
    km.plot_all("save")
