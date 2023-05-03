
import os
from collections import Counter
import sys
import json
import statistics

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
bias = lambda x: x 


def process(subgroup_size, exp):
    filtered_dict = {
        k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items()
    }

    # def __init__(self, timestamp, bssid, distance, ground_truth, ap_location,source='synthetic',exp='EXP_None'):
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
    colors = Util.generate_color_dict_v1(set(cl))
    counter = Counter([x for x in cl if x != -1])
    # find the most common integer
    most_common = -1
    if(len(counter.most_common(1))>0):
        most_common = counter.most_common(1)[0][0]
    return all_pos, cl, colors,most_common


root = tk.Tk()
root.title("Ultra-Wide-Band data static scenario")

# create a figure for the plot
fig = Figure(figsize=(5, 4), dpi=100)

# create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# create a frame for the buttons
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM)

def plot():
    exp = entry_exp.get()
    all_pos, cl, colors,most_common = process(int(entry_subgroup.get()), entry_exp.get())
    fig.clf()
    # create a plot with the line equation y = -x + 5 (green)
    ax=None
    if(not is_2D.get()):
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim([0, 10])
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    for i, point in enumerate(all_pos):
        if cl[i] != -1:
            if(not is_2D.get()):
                ax.scatter(point["x"], point["y"], point["z"], c=colors[cl[i]], marker="x")
            else:
                ax.scatter(point["x"], point["y"],s=50, c=colors[cl[i]], marker="x")
    if(not is_2D):
        ax.scatter(
            mobile_location_dict[exp][0],
            mobile_location_dict[exp][1],
            mobile_location_dict[exp][2],
            c="black",
            marker="o",
        )
    else:
        ax.scatter(
            mobile_location_dict[exp][0],
            mobile_location_dict[exp][1],
            s=50,
            c="black",
            marker="o",
        )
    filtered_points=[]
    print("most_common: ",most_common)
    for i in cl:
        if(cl[i]==most_common):
          filtered_points.append(all_pos[i])

    
# extract the x, y, and z coordinates of the points
    x = [p['x'] for p in filtered_points]
    y = [p['y'] for p in filtered_points]
    z = [p['z'] for p in filtered_points]

# calculate the mean point
    mean_point = {'x': np.mean(x), 'y': np.mean(y), 'z': np.mean(z)}

# calculate the quartiles
    q1_point = {'x': np.percentile(x, 25), 'y': np.percentile(y, 25), 'z': np.percentile(z, 25)}
    q2_point = {'x': np.percentile(x, 50), 'y': np.percentile(y, 50), 'z': np.percentile(z, 50)}
    q3_point = {'x': np.percentile(x, 75), 'y': np.percentile(y, 75), 'z': np.percentile(z, 75)}
    if(not is_2D):
        ax.scatter(q1_point["x"], q1_point["y"], q1_point["z"], c='r', marker="o")
        ax.scatter(q2_point["x"], q2_point["y"], q2_point["z"], c='pink', marker="o")
        ax.scatter(q3_point["x"], q3_point["y"], q3_point["z"], c='blue', marker="o")
    else:
        ax.scatter(q1_point["x"], q1_point["y"], c='r', marker="o")
        ax.scatter(q2_point["x"], q2_point["y"], c='pink', marker="o")
        ax.scatter(q3_point["x"], q3_point["y"], c='blue', marker="o")
    std_dev = {'x': np.std(x), 'y': np.std(y), 'z': np.std(z)}
# print the results
    print(f"Mean point: {mean_point}")
    print(f"1st quartile point: {q1_point}")
    print(f"2nd quartile point: {q2_point}")
    print(f"3rd quartile point: {q3_point}")
    print(f"Standard deviation: {std_dev}")
    # update the canvas with the new plot
    canvas.draw()
button = tk.Button(controls_frame, text="Plot", command=plot)
button.pack(side=tk.LEFT)

# create an input text box
entry_subgroup = tk.Entry(controls_frame)
entry_subgroup.insert(0, 10)
entry_subgroup.pack(side=tk.LEFT)


# create an input text box
entry_exp = tk.Entry(controls_frame)
entry_exp.insert(0, "EXP_1")
entry_exp.pack(side=tk.LEFT)
# start the main event loop

# create a variable to store the state of the checkbox
is_2D = tk.BooleanVar()

# create a checkbox and associate it with the variable
checkbox = tk.Checkbutton(root, text="2D", variable=is_2D)
checkbox.pack()

root.mainloop()
