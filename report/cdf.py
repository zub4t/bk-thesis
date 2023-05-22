import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import cumfreq
import statistics
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
from GradientDescentFixedZ import GradientDescent
from Measurement import Measurement
from ParticleFilter import ParticleFilter
from DistanceAnalyzer import DistanceAnalyzer
from ClusterAnalyzer import ClusterAnalyzer
from KalmanFilter import KalmanFilter
import Util


measurements_dict, mobile_location_dict = Util.read_json_file(
    "../JSON/new_file.json", "802.11mc"
)

with open("../JSON/AP_location.json", "r") as f:
    ap_location_raw = json.load(f)
ap_locations = {}
for e in ap_location_raw:
    ap_locations[e["BSSID"]] = {"x": e["X"], "y": e["Y"], "z": e["Z"]}
gradient_descent = GradientDescent(
    learning_rate=0.01, max_iterations=1000, tolerance=1e-5
)
F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # state transition matrix
H = np.eye(3)  # observation matrix
kf = KalmanFilter(F=F, H=H)

particle_filter = ParticleFilter(initial_particles=np.random.rand(1000, 3), 
                                 process_noise_std=0.1, 
                                 measurement_noise_std=0.1)

all_pos_distance_analyzer = []
all_pos_cluster_analyzer = []
d_c = []
d_d = []
d_c_pf = []
d_d_pf = []
d_d_kf = []
d_c_kf = []
d_a= []
def smooth_current_point_kf(old_points, current_point, kf):
    # Convert points to numpy arrays.
    old_points_array = np.array([[p['x'], p['y'], p['z']] for p in old_points])
    current_point_array = np.array([current_point['x'], current_point['y'], current_point['z']])

    # Update the filter with the old points.
    for point in old_points_array:
        kf.predict()
        kf.update(point)

    # Update the filter with the current point and get the estimate.
    kf.predict()
    kf.update(current_point_array)
    smoothed_current_point_array = kf.x

    # Convert the smoothed current point back to a dictionary.
    smoothed_current_point = {'x': smoothed_current_point_array[0][0], 'y': smoothed_current_point_array[1][0], 'z': smoothed_current_point_array[2][0]}

    return smoothed_current_point

def smooth_current_point_pf(old_points, current_point, particle_filter):
    # Convert points to numpy arrays.
    old_points_array = np.array([[p['x'], p['y'], p['z']] for p in old_points])
    current_point_array = np.array([current_point['x'], current_point['y'], current_point['z']])

    # Update the filter with the old points.
    for point in old_points_array:
        particle_filter.predict()
        particle_filter.update(point)

    # Update the filter with the current point and get the estimate.
    particle_filter.predict()
    particle_filter.update(current_point_array)
    smoothed_current_point_array = particle_filter.estimate()

    # Convert the smoothed current point back to a dictionary.
    smoothed_current_point = {'x': smoothed_current_point_array[0], 'y': smoothed_current_point_array[1], 'z': smoothed_current_point_array[2]}

    return smoothed_current_point
def process(identifier,exp):
    bias = lambda x: x 
    if(identifier=='BSSID'):
        bias = lambda x: x / 1.16 - 0.63
    filtered_dict = {

        k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items()
    }
    averages_dict = {
        k: statistics.mean([obj.distance for obj in v])
        for k, v in filtered_dict.items()
    }


    subgroup_list = Util.generate_subgroups(
        4, arr=list(ap_locations.keys())
    )
    all_pos = []
    gt=mobile_location_dict[exp]
    gt={'x':gt[0],'y':gt[1],'z':gt[2]}
    s = set()
    for i, subgroup in enumerate(subgroup_list):
        measurements = []
        for ap in subgroup:
            measurement = filtered_dict[ap][0]
            measurement.distance = averages_dict[ap]
            measurement.distance = bias(measurement.distance)
            s.add(measurement)
            measurements.append(measurement)
        position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
        all_pos.append(position)

    distanceAnalyzer = DistanceAnalyzer(all_pos)
    point_distance = distanceAnalyzer.get_min_distance_point()
    smoothed_current_point_distance_pf = smooth_current_point_pf(all_pos_distance_analyzer,point_distance, particle_filter)
    smoothed_current_point_distance_kf = smooth_current_point_kf(all_pos_distance_analyzer,point_distance, kf)
    all_pos_distance_analyzer.append(point_distance)
#----------------------------------------------------------------------------------------------------------------
    clusterAnalyzer = ClusterAnalyzer(all_pos)
    point_to_add = clusterAnalyzer.get_mean_of_largest_cluster()
    point_cluster={'x':point_to_add[0],'y':point_to_add[1],'z':point_to_add[2]}
    smoothed_current_point_cluster_pf = smooth_current_point_pf(all_pos_cluster_analyzer,point_cluster, particle_filter)
    smoothed_current_point_cluster_kf = smooth_current_point_kf(all_pos_cluster_analyzer,point_cluster, kf)
    all_pos_cluster_analyzer.append(point_cluster)

    d_c.append(Util.calculate_distance(point_cluster,gt))
    d_d.append(Util.calculate_distance(point_distance,gt))
    d_c_pf.append(Util.calculate_distance(smoothed_current_point_cluster_pf,gt))
    print(smoothed_current_point_cluster_kf)
    d_c_kf.append(Util.calculate_distance(smoothed_current_point_cluster_kf,gt))
    d_d_pf.append(Util.calculate_distance(smoothed_current_point_distance_pf,gt))
    d_d_kf.append(Util.calculate_distance(smoothed_current_point_distance_kf,gt))
    
    position = gradient_descent.train(list(s), {"x": 0, "y": 0, "z": 0})
    d_a.append(Util.calculate_distance(position,gt))
    
if __name__ == "__main__":
    for i in range(1, 46):
        exp = f'EXP_{i}'
        process('BBSID', exp)
        print('end ',i)
    Util.create_cdf_plot(d_c, 'CDF_distance_cluster.png')
    Util.create_cdf_plot(d_d, 'CDF_distance_distance.png')
    Util.create_cdf_plot(d_c_pf, 'CDF_disctance_cluster_pf.png')
    Util.create_cdf_plot(d_d_pf, 'CDF_distance_distance_pf.png')
    Util.create_cdf_plot(d_a, 'CDF_distance_all.png')
    Util.create_cdf_plot(d_c_kf, 'CDF_disctance_cluster_kf.png')
    Util.create_cdf_plot(d_d_kf, 'CDF_distance_distance_kf.png')

