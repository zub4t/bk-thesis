import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import cumfreq
import statistics
import sys
import os
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.stats import cumfreq
from sklearn.preprocessing import MinMaxScaler

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
# Assuming these are your custom classes
from GradientDescent import GradientDescent
from Measurement import Measurement
import Util
# from pykalman import KalmanFilter
#
# class MyKalmanFilter:
#     def __init__(self, initial_state, initial_state_covariance):
#         self.kf = KalmanFilter(initial_state_mean=initial_state,
#                                initial_state_covariance=initial_state_covariance)
#     def update(self, observed_points):
#         # Assume the state transition is identity and observation model is identity
#         # This means the state does not change over time and the observed point is the state
#         # Also assume the noise covariances are identity matrices
#         # These assumptions may not be valid for your specific problem
#         transition_matrix = np.eye(3)
#         observation_matrix = np.eye(3)
#         transition_covariance = np.eye(3)
#         observation_covariance = np.eye(3)
#
#         for observed_point in observed_points:
#             # Convert the observed point from a dictionary to a 1D array
#             observed_point_array = np.array([observed_point['x'], observed_point['y'], observed_point['z']])
#
#             (state_mean, state_covariance) = self.kf.filter_update(
#                 self.kf.initial_state_mean,
#                 self.kf.initial_state_covariance,
#                 observation=observed_point_array,
#                 transition_matrix=transition_matrix,
#                 observation_matrix=observation_matrix,
#                 transition_covariance=transition_covariance,
#                 observation_covariance=observation_covariance
#             )
#
#         self.kf.initial_state_mean = state_mean
#         self.kf.initial_state_covariance = state_covariance
#     def get_state(self):
#         return self.kf.initial_state_mean

class ParticleFilter:
    def __init__(self, num_particles, space_constraints, motion_model, measurement_model):
        self.num_particles = num_particles
        self.space_constraints = space_constraints
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            particle = {
                'x': random.uniform(self.space_constraints['x_min'], self.space_constraints['x_max']),
                'y': random.uniform(self.space_constraints['y_min'], self.space_constraints['y_max']),
                'z': random.uniform(self.space_constraints['z_min'], self.space_constraints['z_max']),
                'weight': 1.0 / self.num_particles
            }
            particles.append(particle)
        return particles

    def _predict_particles(self):
        for particle in self.particles:
            particle['x'] += self.motion_model['velocity_x']
            particle['y'] += self.motion_model['velocity_y']
            particle['z'] += self.motion_model['velocity_z']

    def _update_weights(self, observed_points):
        for particle in self.particles:
            particle['weight'] = 1.0
            for observed_point in observed_points:
                likelihood = self.measurement_model(particle, observed_point)
                particle['weight'] *= likelihood

        total_weight = sum(particle['weight'] for particle in self.particles)
        for particle in self.particles:
            particle['weight'] /= (total_weight + 1e-9)

    def _resample_particles(self):
        weights = [particle['weight'] for particle in self.particles]
        total_weight = sum(weights)
        if(total_weight == 0 ):
            total_weight+= + 1e-9
        normalized_weights = [weight / total_weight for weight in weights]
        cum_weights = [sum(normalized_weights[:i+1]) for i in range(len(normalized_weights))]

        new_particles = []
        for _ in range(self.num_particles):
            rand_num = random.uniform(0, 1)
            for i, cum_weight in enumerate(cum_weights):
                if rand_num <= cum_weight:
                    new_particles.append(self.particles[i].copy())
                    break
        self.particles = new_particles

    def update(self, observed_points):
        self._update_weights(observed_points)
        self._resample_particles()

    def get_most_likely_particle(self):
        max_weight = 0.0
        most_likely_particle = None

        for particle in self.particles:
            if particle['weight'] > max_weight:
                max_weight = particle['weight']
                most_likely_particle = particle

        return most_likely_particle

# Step 1: From Distance Measurements To Position
def distance_to_position(averages_dict, initial_position, gradient_descent, subgroup_list):
    positions = []
    for subgroup in subgroup_list:
        subgroup_averages = [ averages_dict[k] for k in subgroup]
        position = gradient_descent.train(subgroup_averages, initial_position)
        positions.append(position)
    return positions


# Step 2: From A Constellation Of Points To A Single One
def constellation_to_single_point(points):
    # DBSCAN Clustering and Mean Calculation
    # Convert list of dictionaries to 2D array
    points_2d = np.array([[point['x'], point['y'], point['z']] for point in points])

    # DBSCAN Clustering and Mean Calculation
    clustering = DBSCAN(eps=3, min_samples=2).fit(points_2d)
    labels = clustering.labels_
    largest_cluster_index = np.argmax(np.bincount(labels[labels != -1]))
    largest_cluster = points_2d[labels == largest_cluster_index]
    mean_point = np.mean(largest_cluster, axis=0)

    # Minimum Sum of Distances
    distances = distance.cdist(points_2d, points_2d)
    sum_distances = distances.sum(axis=1)
    min_sum_point = points_2d[sum_distances.argmin()]

    def measurement_model(particle, observed_point):
        distance = math.sqrt((particle['x'] - observed_point['x'])**2 +
                             (particle['y'] - observed_point['y'])**2 +
                             (particle['z'] - observed_point['z'])**2)
        likelihood = math.exp(-distance)
        return likelihood

    num_particles = 100
    space_constraints = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 6, 'z_min': 0, 'z_max': 4}
    motion_model = {'velocity_x': 0, 'velocity_y': 0, 'velocity_z': 0}
    particle_filter = ParticleFilter(num_particles, space_constraints, motion_model, measurement_model)
    particle_filter.update(points)
    particle_filter_point = particle_filter.get_most_likely_particle()
    # particle filter given none ?
    if particle_filter_point != None:
        particle_filter_point = np.array([particle_filter_point['x'],particle_filter_point['y'],particle_filter_point['z']])
        #kalman_filter_point = kf.estimate()
        print("particle filter ",particle_filter_point)
        return [particle_filter_point]#[mean_point]#, min_sum_point,particle_filter_point]#, kalman_filter_point
    else:
        return  [[10,10,10]]#[mean_point, min_sum_point] 
# Step 3: Mean of Single Points
def mean_of_single_points(points):
    print(points)
    return np.mean(points, axis=0)

def create_cdf(data, filename):
    # Create CDF
    a = np.array(data)
    num_bins = 20
    counts, bin_edges = np.histogram(a, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf)
    plt.savefig(filename)

def main():
    # Prompt user for technology choice
    tech_choice = input("Please enter the technology choice (uwb or 802.11mc): ")
    os.makedirs(tech_choice, exist_ok=True)

    # Load your data
    measurements_dict, mobile_location_dict = Util.read_json_file(
        "../JSON/new_file.json", tech_choice
    )

    # Initialize your measurements and positions
    measurements = []  # Load your measurements
    initial_position = {"x": 0, "y": 0, "z": 0}  # Define your initial position
    gradient_descent = GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)

    # Bias correction for 802.11mc
    identifier = 'BSSID' if tech_choice == '802.11mc' else 'ID'
    def bias(x):
        if identifier == 'BSSID':
            return x / 1.16 - 0.63
        else:
            return x
    diff = []
    # Loop over all experiments
    for exp in range(1, 45):  # Assuming there are 10 experiments

        gt=mobile_location_dict[f'EXP_{exp}']
        gt={'x':gt[0],'y':gt[1],'z':gt[2]}
        print("working on ",exp)
        # Filter measurements for the current experiment
        filtered_dict = {
            k: [obj for obj in v if obj.exp == f'EXP_{exp}'] for k, v in measurements_dict.items()
        }
        # Calculate average distances for the current experiment
        averages_dict = {
            k: statistics.mean([bias(obj.distance) for obj in v])
            for k, v in filtered_dict.items()
        }

        averages_dict = {
            k: 
            Measurement(
                    v[0].timestamp,
                    v[0].bssid,
                    statistics.mean([bias(obj.distance) for obj in v]),
                    0,
                    v[0].ap_location
                    )
            for k, v in filtered_dict.items()
        }
        # Generate subgroups
        subgroup_list = Util.generate_subgroups(len(averages_dict.keys()), arr=list(averages_dict.keys()))

        # Step 1
        positions = distance_to_position(averages_dict, initial_position, gradient_descent,subgroup_list)

        # Step 2
        #single_points = constellation_to_single_point(positions)

        # Step 3
        # final_position = mean_of_single_points(single_points)
        # final_position = {'x':final_position[0],'y':final_position[1],'z':final_position[2]}
        final_position = positions[0]
        df =Util.calculate_distance(gt,final_position)
        print('df',df)
        print('final position vs gt ',final_position,gt)
        diff.append(df)
    Util.create_cdf_plot(diff, 'CDF_all.png')
if __name__ == "__main__":
    main()

