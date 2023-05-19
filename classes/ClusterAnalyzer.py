from sklearn.cluster import DBSCAN
import numpy as np

class ClusterAnalyzer:
    def __init__(self, points, eps=3, min_samples=2):
        self.points = points
        self.eps = eps
        self.min_samples = min_samples

    def get_coords(self):
        # Extract point coordinates from dictionaries
        return [list(point.values()) for point in self.points]

    def perform_clustering(self):
        # Perform DBSCAN clustering
        self.clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.get_coords())

    def get_labels(self):
        # Get cluster labels
        return self.clustering.labels_

    def get_largest_cluster_label(self):
        # Get the label of the largest cluster
        return max(set(self.get_labels()), key=list(self.get_labels()).count)

    def get_largest_cluster_coords(self):
        # Extract the coordinates of points in the largest cluster
        return [coord for coord, label in zip(self.get_coords(), self.get_labels()) if label == self.get_largest_cluster_label()]

    def get_mean_of_largest_cluster(self):
        # Calculate and return the mean of the points in the largest cluster
        self.perform_clustering()
        return np.mean(self.get_largest_cluster_coords(), axis=0)

