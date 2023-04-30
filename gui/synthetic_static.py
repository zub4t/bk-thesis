import os
import sys

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))
import Util
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN
import numpy as np
from Synthetic import Synthetic
from GradientDescent import GradientDescent

# measurements_dict, mobile_location_dict = Util.read_json_file("../file.json")
# create the main window
gradient_descent = GradientDescent(
    learning_rate=0.01, max_iterations=1000, tolerance=1e-5
)
synthetic = Synthetic()
ap_locations = synthetic.generate_random_ap_location(12)
def process(subgroup_size):
    subgroup_list = Util.generate_subgroups(subgroup_size, arr=list(ap_locations.keys()))
    measurements_dict = synthetic.generate_synthetic_data_static(12,ap_locations=ap_locations)
    print(measurements_dict)
    all_pos = []
    for i, subgroup in enumerate(subgroup_list):
        measurements = []
        for ap in subgroup:
                measurement = measurements_dict[ap][0]
                measurements.append(measurement)
        position = gradient_descent.train(
            measurements, {"x": 0, "y": 0, "z": 0}
        )
        all_pos.append(position)
    l = [ [p["x"],p["y"],p["z"]] for p in all_pos]
    dbscan = DBSCAN(eps=.30, min_samples=4)
    dbscan.fit(np.array(l))
    cl = (dbscan.labels_)
    colors = Util.generate_color_dict(set(cl))
    return all_pos,cl,colors
root = tk.Tk()
root.title("Synthetic data static scenario")

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
    all_pos,cl,colors = process(int(entry.get()))
    fig.clf()
    # create a plot with the line equation y = -x + 5 (green)
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for i,point in enumerate(all_pos):
        if(cl[i]!=-1):
            ax.scatter(point["x"], point["y"], point["z"], c=colors[cl[i]], marker="o")
    
    ax.scatter(synthetic.static_ground_truth["x"], synthetic.static_ground_truth["y"], synthetic.static_ground_truth["z"], c="black", marker="o")
    # update the canvas with the new plot
    canvas.draw()


button = tk.Button(controls_frame, text="Plot", command=plot)
button.pack(side=tk.LEFT)

# create an input text box
entry = tk.Entry(controls_frame)
entry.insert(0,10)
entry.pack(side=tk.LEFT)


# start the main event loop
root.mainloop()
