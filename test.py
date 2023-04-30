from sklearn.cluster import DBSCAN
import numpy as np
import threading
import math
import os
import sys
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "classes"))
sys.path.insert(0, os.path.join(cwd, "commons"))
from APLocation import APLocation
from GradientDescent import GradientDescent
from GradientDescentFixedZ import GradientDescentFixedZ
from Measurement import Measurement
from Synthetic import Synthetic
from Util import (
    generate_intermediate_points,
    generate_person_path,
    generate_subgroups,
    group_measurements_by_bssid,
    interpolate_from_timestamp_to_location,
    interpolate_points,
    read_timestamps,
    calculate_distance,
    append_to_file,
    generate_color_dict,
)
def process_data(bias_func):
    measurements = Measurement.read_json_file("./new_file.json", "EXP_42", "802.11mc")
    measurements_dict = group_measurements_by_bssid(measurements)
    gradient_descent = GradientDescent(
        learning_rate=0.01, max_iterations=1000, tolerance=1e-5
    )
    print((measurements_dict.keys()))
    subgroup_list = generate_subgroups(4, arr=list(measurements_dict.keys()))
    all_pos = []
    ground_truth = None
    for i, subgroup in enumerate(subgroup_list):
        measurements = []
        for ap in subgroup: 
                measurement = measurements_dict[ap][1]
                measurement.distance = bias_func(measurement.distance)
                ground_truth = measurement.ground_truth
                measurements.append(measurement)
        position = gradient_descent.train(measurements, {"x": 0, "y": 0, "z": 0})
        all_pos.append(position)

    l = [[element["x"], element["y"], element["z"]] for element in all_pos]
    X = np.array(l)

# instantiate DBSCAN object with epsilon=1.0 and min_samples=2
    dbscan = DBSCAN(eps=0.30, min_samples=2)

# fit the DBSCAN model to the data
    dbscan.fit(X)

# print the cluster labels (-1 indicates noise)
    arr_label = dbscan.labels_
    color_dict = generate_color_dict(set(arr_label))
    return all_pos,color_dict,arr_label

# create the main window
root = tk.Tk()
root.title("Line Plotter")

# create a figure for the plot
fig = Figure(figsize=(5, 4), dpi=100)

# create a canvas to display the plot
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# create a frame for the sliders, button, and dropdown
controls_frame = tk.Frame(root)
controls_frame.pack(side=tk.BOTTOM)

# create a label for the slope slider
slope_label = tk.Label(controls_frame, text="Slope (m):")
slope_label.pack(side=tk.LEFT)

# create a slider for the slope
slope_slider = tk.Scale(controls_frame, from_=-5, to=5, resolution=0.1, orient=tk.HORIZONTAL)
slope_slider.pack(side=tk.LEFT)

# create a label for the intercept slider
intercept_label = tk.Label(controls_frame, text="Intercept (b):")
intercept_label.pack(side=tk.LEFT)

# create a slider for the intercept
intercept_slider = tk.Scale(controls_frame, from_=-10, to=10, resolution=0.1, orient=tk.HORIZONTAL)
intercept_slider.pack(side=tk.LEFT)

# create a dropdown to select 2D or 3D plot type
plot_type_var = tk.StringVar(value="2D")
plot_type_dropdown = tk.OptionMenu(controls_frame, plot_type_var, "2D", "3D")
plot_type_dropdown.pack(side=tk.LEFT)

# create a function to plot the line
def plot_line():
    # clear the previous plot
    fig.clf()

    # get the slope and intercept values from the sliders
    m = slope_slider.get()
    b = intercept_slider.get()
    if(m==0):
        m=0.1
    bias_func = lambda x: x//(m) - b
    # get the selected plot type from the dropdown
    plot_type = plot_type_var.get()

    if plot_type == "2D":
        # create a 2D plot with the line equation y = m*x + b
        ax = fig.add_subplot(111)
        x = [-10, 10]
        y = [m*x_i + b for x_i in x]
        ax.plot(x, y)
    elif plot_type == "3D":
        all_pos,color_dict,arr_label = process_data(bias_func)
        ax = fig.add_subplot(111, projection='3d')
        for i,d in enumerate(all_pos):
            x, y, z = d["x"], d["y"], d["z"]
            ax.scatter(x, y, z,c=color_dict[arr_label[i]])
        ax.scatter(8.5, 0.5, 0,c='black',marker='x')
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # update the canvas with the new plot
    canvas.draw()

# create a button to plot the line
plot_button = tk.Button(controls_frame, text="Plot", command=plot_line)
plot_button.pack(side=tk.LEFT)

# start the main event loop
root.mainloop()

