import os
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
from GradientDescent import GradientDescent
from Measurement import Measurement
colors=None
measurements_dict, mobile_location_dict = Util.read_json_file(
    "../JSON/file.json", "802.11mc"
)
print(measurements_dict)
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


def process(subgroup_size, exp):

    filtered_dict = {
        k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items() if any(obj.exp == exp for obj in v)
    }
    sampled_list = random.sample(filtered_dict.keys(), 6)
    subgroup_list = Util.generate_subgroups(subgroup_size, arr=sampled_list)
    all_pos = {}
    return filtered_dict, subgroup_list
    cl = all_pos.keys()


root = tk.Tk()
root.title("WiFi path 4 of 5")

# create a figure for the plot
fig = Figure(figsize=(5, 4), dpi=100)

# create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# create a frame for the buttons
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM)

fig.clf()
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
real_person_path = Util.interpolate_points(Measurement.points_exp, 20)

x = [point["x"] for point in real_person_path]
y = [point["y"] for point in real_person_path]
z = [point["z"] for point in real_person_path]
ax.scatter(x, y, z, c="r", marker="o")
def plot():
    global Colors
    global Subgroup_list 
    global Filtered_dict
    global Index
    all_pos = {}

    for i, subgroup in enumerate(Subgroup_list):
        measurements = []
        for ap in subgroup:
            try:
                measurement = Filtered_dict[ap].pop(0)
                measurement.distance = bias(measurement.distance)
                measurements.append(measurement)
            except:
                print("e")
        position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
        if subgroup in all_pos:
            all_pos[subgroup].append(position)
        else:
            all_pos[subgroup] = [position]
    for i,key in enumerate(all_pos):
        point = all_pos[key][0]
        ax.scatter(point['x'], point['y'],point['z'], s=20, c=Colors[key], marker="x")
    Index+=1    
    canvas.draw()


button = tk.Button(controls_frame, text="Plot", command=plot)
button.pack(side=tk.LEFT)

# create an input text box
entry_exp = tk.Entry(controls_frame)
entry_exp.insert(0, "EXP_74")
entry_exp.pack(side=tk.LEFT)
# start the main event loop
Filtered_dict, Subgroup_list = process(4, entry_exp.get())
Colors = None
Colors = Util.generate_color_dict_v1(set(Subgroup_list))
Index = 0

root.mainloop()
