import os
import subprocess
from PIL import Image, ImageTk
from collections import Counter
import sys
import json
import statistics
import random

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
import Util
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN
import numpy as np
from GradientDescentFixedZ import GradientDescent
from Measurement import Measurement
colors=None
measurements_dict, mobile_location_dict = Util.read_json_file(
    "../JSON/file.json", "uwb"
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


def process(subgroup_size, exp):

    filtered_dict = {
        k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items() if any(obj.exp == exp for obj in v)
    }

    filtered_dict = Util.filter_measurements(filtered_dict)
    sampled_list = list(filtered_dict.keys())
    subgroup_list = Util.generate_subgroups(subgroup_size, arr=sampled_list)
    return filtered_dict, subgroup_list,sampled_list
    #cl = all_pos.keys()
root = tk.Tk()
root.attributes("-fullscreen", True)
root.title("WiFi path 6 of 4")
# create a figure for the plot
fig = Figure(figsize=(6, 4), dpi=100)

# create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
#canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
# create a frame for the buttons
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM)

def plot():

    fig.clf()
    ax = fig.add_subplot(111) 
    ax.set_xlim([-1, 10])
    ax.set_ylim([-5, 10])
   # ax.set_zlim([0, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    #ax.set_zlabel("Z")
    real_person_path = Util.interpolate_points(Measurement.points_exp, 20)

    x = [point["x"] for point in real_person_path]
    y = [point["y"] for point in real_person_path]
    #z = [point["z"] for point in real_person_path]
    #ax.scatter(x, y, z, c="r", marker="o")
    ax.scatter(x, y, c="r", marker="o")
    global Colors
    global Subgroup_list 
    global Filtered_dict
    global Index
    global Timestamp_list
    global Points_list
    global Sampled_list

    m = []
    for ap in Filtered_dict.keys():
        measurement = Filtered_dict[ap][0]
        measurement.distance = bias(measurement.distance)
        m.append(measurement)

    pos_using_all = gradient_descent.train(m, {"x": 0, "y": 0, "z": 0})
    ax.scatter(pos_using_all['x'], pos_using_all['y'], s=20, c='green', marker="x")
    all_pos = []
    all_gt=[]
    for i, subgroup in enumerate(Subgroup_list):
        measurements = []
        for ap in subgroup:
            try:
                measurement = Filtered_dict[ap][0]
                measurement.distance = bias(measurement.distance)
                measurement.ground_truth = (
                                           Util.interpolate_from_timestamp_to_location(
                                                Points_list,
                                                Timestamp_list,
                                                measurement.timestamp,
                                            )
                                        )
                measurements.append(measurement)
            except:
                print("e")
        position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
        min_timestamp = min(m.timestamp for m in measurements)
        max_timestamp = max(m.timestamp for m in measurements)
        time_diff_milliseconds = max_timestamp - min_timestamp
        time_diff_seconds = time_diff_milliseconds / 1000

        print(f"Time difference in milliseconds: {time_diff_seconds}")
#        if time_diff_seconds <=1:
        all_gt.append(measurements[0].ground_truth)
        all_pos.append(position)
    for ap in Sampled_list: 
        Filtered_dict[ap].pop(0)

    min_point = Util.min_sum_distances_points(all_pos)
    ax.scatter(min_point['x'], min_point['y'], s=20, c='black', marker="x")

    dbscan = DBSCAN(eps=0.10, min_samples=4) 
    l = [[p["x"], p["y"]] for p in all_pos]
    dbscan.fit(np.array(l))
    cl = dbscan.labels_
    points_by_cluster = {}
    for i, point in enumerate(all_pos):
        if cl[i] != -1:
            if cl[i] in points_by_cluster:
                points_by_cluster[cl[i]].append(point)
            else:
                points_by_cluster[cl[i]] = [point]
    
    most_populated_key = max(points_by_cluster, key=lambda x: len(points_by_cluster[x]))
    mean_point = Util.calculate_mean_point(points_by_cluster[most_populated_key])
    ax.scatter(mean_point['x'], mean_point['y'], s=20, c='red', marker="x")
    for gt in all_gt:
        ax.scatter(gt['x'], gt['y'], s=20, c='b', marker="o")
    Index+=1    
    canvas.draw()
    filename = "plots_comparison/plot_{}.png".format(Index)  # include the counter variable in the filename
    fig.savefig(filename)

button = tk.Button(controls_frame, text="Plot", command=plot)
button.pack(side=tk.LEFT)

# create an input text box
entry_exp = tk.Entry(controls_frame)
entry_exp.insert(0, "EXP_56")
entry_exp.pack(side=tk.LEFT)
# start the main event loop
Timestamp_list = Util.read_timestamps(
            f"../CHECKPOINTS/CHECKPOINT_EXP_73"
        )
Filtered_dict, Subgroup_list,Sampled_list = process(4, entry_exp.get())
Colors = None
Colors = Util.generate_color_dict_v1(set(Subgroup_list))
Index = 0
Points_list = Util.generate_intermediate_points(Measurement.points_exp)
#root.mainloop()
while(True):
    plot()
