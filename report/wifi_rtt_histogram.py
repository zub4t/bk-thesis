import os
import sys
import json
import statistics
import matplotlib.pyplot as plt
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
import Util
from sklearn.cluster import DBSCAN
import numpy as np
from GradientDescent import GradientDescent
from Measurement import Measurement


measurements_dict, mobile_location_dict = Util.read_json_file(
    "../JSON/new_file.json", "uwb"
)
with open("../JSON/AP_location.json", "r") as f:
    ap_location_raw = json.load(f)
ap_locations = {}
for e in ap_location_raw:
    ap_locations[e["ID"]] = {"x": e["X"], "y": e["Y"], "z": e["Z"]}
# create the main window
gradient_descent = GradientDescent(
    learning_rate=0.01, max_iterations=1000, tolerance=1e-5
)
bias = lambda x: x #/ 1.16 - 0.63
all_min_diff = []
all_using_all_m_diff = []
all_using_mean_cluster = []
def main(exp_target, subgroup_size): 
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
        pos_by_ap = {}
        pos_using_all = None
        gt = {
            "x": mobile_location_dict[exp][0],
            "y": mobile_location_dict[exp][1],
            "z": mobile_location_dict[exp][2],
        }
        m = []
        for ap in filtered_dict.keys():
            measurement = filtered_dict[ap][0]
            measurement.distance = averages_dict[ap]
            measurement.distance = bias(measurement.distance)
            m.append(measurement)

        all_using_all_m_diff.append(Util.calculate_distance(gt,gradient_descent.train(m, {"x": 0, "y": 0, "z": 0})))
        for i, subgroup in enumerate(subgroup_list):
            measurements = []
            for ap in subgroup:
                measurement = filtered_dict[ap][0]
                measurement.distance = averages_dict[ap]
                measurement.distance = bias(measurement.distance)
                measurements.append(measurement)
            position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
            all_pos.append(position)
            for ap in subgroup:
                if ap in pos_by_ap:
                    pos_by_ap[ap].append(Util.calculate_distance(position,gt))
                else:
                    pos_by_ap[ap] = [Util.calculate_distance(position,gt)]
        l = [[p["x"], p["y"], p["z"]] for p in all_pos]
        dbscan = DBSCAN(eps=0.10, min_samples=4)
        dbscan.fit(np.array(l))
        cl = dbscan.labels_
        return all_pos, cl, pos_by_ap

    def log(exp, subgroup):
        all_pos, cl, pos_by_ap = process(subgroup, exp)

        for key, value in pos_by_ap.items():
            # Calculate mean and standard deviation
            mean = np.mean(value)
            stdv = np.std(value)

            # Create histogram
            fig, ax = plt.subplots()
            ax.hist(value, bins="auto")

            # Add labels and title
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of " + key)

            # Add mean and standard deviation to plot
            ax.axvline(x=mean, color="red", label="Mean = {:.2f}".format(mean))
            ax.axvline(
                x=mean + stdv, color="purple", label="Std Dev = {:.2f}".format(stdv)
            )
            ax.axvline(x=mean - stdv, color="purple")

            # Add legend
            ax.legend()
            fig.savefig(f"./histogram/{key}_{exp}_histogram.png")

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

        min_point = Util.calculate_distance(gt, Util.min_sum_distances_points(all_pos))
        most_populated_key = max(points_by_cluster, key=lambda x: len(points_by_cluster[x]))
        mean_point = Util.calculate_mean_point(points_by_cluster[most_populated_key])
        all_using_mean_cluster.append(Util.calculate_distance(gt,mean_point))
        all_min_diff.append(min_point)
    log(exp_target,subgroup_size)


for key in mobile_location_dict:
    print(key)
    main(str(key), 4)
#using the point obtanied in the min sum
plt.clf()
sorted_data = np.sort(all_min_diff)
cumulative_probabilities = np.arange(len(sorted_data)) / float(len(sorted_data))
plt.plot(sorted_data, cumulative_probabilities, marker='o')
# Set the title and axis labels
plt.title("Cumulative Distribution Function")
plt.xlabel("Values")
plt.ylabel("Cumulative Probability")
plt.savefig('CDF_LOWEST_SUM.png')
#using all measurements to obtain the point

plt.clf()
sorted_data = np.sort(all_using_all_m_diff)
cumulative_probabilities = np.arange(len(sorted_data)) / float(len(sorted_data))
plt.plot(sorted_data, cumulative_probabilities, marker='o')
# Set the title and axis labels
plt.title("Cumulative Distribution Function")
plt.xlabel("Values")
plt.ylabel("Cumulative Probability")
plt.savefig('CDF_USING_ALL.png')


plt.clf()
sorted_data = np.sort(all_using_mean_cluster)
cumulative_probabilities = np.arange(len(sorted_data)) / float(len(sorted_data))
plt.plot(sorted_data, cumulative_probabilities, marker='o')
# Set the title and axis labels
plt.title("Cumulative Distribution Function")
plt.xlabel("Values")
plt.ylabel("Cumulative Probability")
plt.savefig('CDF_USING_MEAN_CLUSTER.png')
