import numpy as np
from scipy.spatial import distance

class DistanceAnalyzer:
    def __init__(self, points):
        self.points = points

    def get_coords(self):
        # Extract point coordinates from dictionaries
        return [list(point.values()) for point in self.points]

    def calculate_total_distances(self):
        # Calculate the sum of the distances from each point to all other points
        coords = self.get_coords()
        total_distances = [sum(distance.euclidean(c1, c2) for c2 in coords) for c1 in coords]
        return total_distances

    def get_min_distance_point(self):
        # Return the point that has the minimum total distance
        total_distances = self.calculate_total_distances()
        min_distance_index = total_distances.index(min(total_distances))
        return self.points[min_distance_index]

