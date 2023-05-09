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
    sampled_list = ['5A0A','111F','D018','D713','120F','868C']
    subgroup_list = Util.generate_subgroups(subgroup_size, arr=sampled_list)
    all_pos = {}
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
    all_pos = {}
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
        if subgroup in all_pos:
            all_pos[subgroup].append(position)
        else:
            all_pos[subgroup] = [position]
        for ap in Sampled_list: 
            Filtered_dict[ap].pop(0)
    column=10    
    for i,key in enumerate(all_pos):
        point = all_pos[key][0]
        text = f"{key}"
        y_random = column  
        x_random = (i%4)*2  
        if(i%4 !=0):
            x_random = (i%4)*2  
        else:
            column -=1
        ax.scatter(x_random, y_random)
        ax.plot([x_random, point['x']], [y_random, point['y']],linestyle=":",alpha=0.4)

        text_x = x_random + 0.2  # add a small offset to the x coordinate
        text_y = y_random
        ax.text(text_x, text_y, text,fontsize=8)
        ax.scatter(point['x'], point['y'], s=20, c=Colors[key], marker="x")

    for gt in all_gt:
        ax.scatter(gt['x'], gt['y'], s=20, c='b', marker="o")
        #ax.scatter(point['x'], point['y'],point['z'], s=20, c=Colors[key], marker="x")
    Index+=1    
    canvas.draw()
    filename = "plots/plot_{}.png".format(Index)  # include the counter variable in the filename
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

root.mainloop()
