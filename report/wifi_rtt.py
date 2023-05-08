import csv
import os
import sys
import json
import statistics

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
import Util
from sklearn.cluster import DBSCAN
import numpy as np
from GradientDescent import GradientDescent
from Measurement import Measurement


measurements_dict, mobile_location_dict = Util.read_json_file(
    "../JSON/new_file.json", "802.11mc"
)
with open("../JSON/AP_location.json", "r") as f:
    ap_location_raw = json.load(f)
ap_locations = {}
for e in ap_location_raw:
    ap_locations[e["BSSID"]] = {"x": e["X"], "y": e["Y"], "z": e["Z"]}
# create the main window
gradient_descent = GradientDescent(
    learning_rate=0.01, max_iterations=1000, tolerance=1e-5
)
bias = lambda x: x / 1.16 - 0.63


def main(log_name, exp_target, subgroup_size):
    def process(subgroup_size, exp):
        filtered_dict = {
            k: [obj for obj in v if obj.exp == exp]
            for k, v in measurements_dict.items()
        }
        print("my exp ", exp)
        averages_dict = {
            k: statistics.mean([obj.distance for obj in v])
            for k, v in filtered_dict.items()
        }
        subgroup_list = Util.generate_subgroups(
            subgroup_size, arr=list(ap_locations.keys())
        )
        all_pos = []
        for i, subgroup in enumerate(subgroup_list):
            measurements = []
            for ap in subgroup:
                measurement = filtered_dict[ap][0]
                measurement.distance = averages_dict[ap]
                measurement.distance = bias(measurement.distance)
                measurements.append(measurement)
            position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
            all_pos.append(position)
        l = [[p["x"], p["y"], p["z"]] for p in all_pos]
        dbscan = DBSCAN(eps=0.10, min_samples=4)
        dbscan.fit(np.array(l))
        cl = dbscan.labels_
        return all_pos, cl

    def log(exp, subgroup):
        all_pos, cl = process(subgroup, exp)
        points_by_cluster = {}
        gt = {
            "x": mobile_location_dict[exp][0],
            "y": mobile_location_dict[exp][1],
            "z": mobile_location_dict[exp][2],
        }
        for i, point in enumerate(all_pos):
            if cl[i] != -1:
                if cl[i] in points_by_cluster:
                    points_by_cluster[cl[i]].append(point)
                else:
                    points_by_cluster[cl[i]] = [point]
        all_mean = []
        all_num_elements_in_cluster = []
        all_distance_to_real_loc = []

        for key in points_by_cluster:
            x = [p["x"] for p in points_by_cluster[key]]
            y = [p["y"] for p in points_by_cluster[key]]
            z = [p["z"] for p in points_by_cluster[key]]
            mean_point = {"x": np.mean(x), "y": np.mean(y), "z": np.mean(z)}
            all_mean.append(mean_point)
            all_num_elements_in_cluster.append(len(points_by_cluster[key]))
            all_distance_to_real_loc.append(Util.calculate_distance(gt, mean_point))
            min_point = Util.calculate_distance(gt, Util.min_sum_distances_points(all_pos))

        write_csv(
            exp_target,
            all_num_elements_in_cluster,
            all_mean,
            all_distance_to_real_loc,
            log_name,
            min_point
        )

    def write_csv(
        exp_name,
        num_elements_in_cluster,
        mean_of_which_cluster,
        distance_to_real_loc,
        file_path,
        min_point
    ):
        data = list(
            zip(
                [exp_name] * len(mean_of_which_cluster),
                num_elements_in_cluster,
                mean_of_which_cluster,
                distance_to_real_loc,
                [min_point] * len(mean_of_which_cluster),
            )
        )

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "exp_name",
                    "num_elements_in_cluster",
                    "mean_of_which_cluster",
                    "distance_to_real_loc",
                ]
            )
            writer.writerows(data)

    log(exp_target, subgroup_size)


for key in mobile_location_dict:
    print(key)
    main(f"./wifi_4/report_{key}_group_size_4.csv", str(key), 4)
