import os
import sys
import json
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN
import numpy as np
import random
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "../classes"))
sys.path.insert(0, os.path.join(cwd, "../commons"))

from GradientDescentFixedZ import GradientDescent
from Measurement import Measurement
from ParticleFilter import ParticleFilter
from DistanceAnalyzer import DistanceAnalyzer
from ClusterAnalyzer import ClusterAnalyzer
import Util
# Initialize the Particle Filter with random particles.
particle_filter = ParticleFilter(initial_particles=np.random.rand(1000, 3), 
                                 process_noise_std=0.1, 
                                 measurement_noise_std=0.1)
all_pos_distance_analyzer = []
all_pos_cluster_analyzer = []
def smooth_current_point(old_points, current_point, particle_filter):
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
class WiFiPathPlotter:
    def __init__(self, root, fig, canvas, controls_frame, entry_exp):
        self.root = root
        self.fig = fig
        self.canvas = canvas
        self.controls_frame = controls_frame
        self.entry_exp = entry_exp
        self.colors = None
        self.index = 0
        self.filtered_dict, self.subgroup_list, self.sampled_list = self.process(4, entry_exp.get())
        self.timestamp_list = Util.read_timestamps(f"../CHECKPOINTS/CHECKPOINT_EXP_73")
        self.points_list = Util.generate_intermediate_points(Measurement.points_exp)
        self.colors = Util.generate_color_dict_v1(set(self.subgroup_list))
        self.gradient_descent = GradientDescent(
            learning_rate=0.01, max_iterations=1000, tolerance=1e-5
        )
        self.bias = lambda x: x# / 1.16 - 0.63

    def process(self, subgroup_size, exp):
        measurements_dict, mobile_location_dict = Util.read_json_file("../JSON/file.json", "uwb")
        filtered_dict = {
            k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items() if any(obj.exp == exp for obj in v)
        }
        sampled_list = (filtered_dict.keys())
        subgroup_list = Util.generate_subgroups(subgroup_size, arr=sampled_list)
        return filtered_dict, subgroup_list, sampled_list

    def plot(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_xlim([-1, 10])
        ax.set_ylim([-5, 10])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        real_person_path = Util.interpolate_points(Measurement.points_exp, 20)
        x = [point["x"] for point in real_person_path]
        y = [point["y"] for point in real_person_path]
        ax.scatter(x, y, c="r", marker="o",s=10)
        measurement_buckets = Util.bucket_measurements(self.filtered_dict, 100)
        print(measurement_buckets)
        try:
            measurements = next(measurement_buckets)
            self.process_measurements(measurements, ax)
        except StopIteration:
            print("No more data")
        self.plot() 

    def process_measurements(self, measurements, ax):
        for key, measurement in measurements.items():
            measurement.distance = self.bias(measurement.distance)
            measurement.ground_truth = (
                Util.interpolate_from_timestamp_to_location(
                    self.points_list,
                    self.timestamp_list,
                    measurement.timestamp,
                )
            )

        all_pos = []
        for i, subgroup in enumerate(self.subgroup_list):
            subgroup_measurements = []
            include_point = True
            for ap in subgroup:
                try:
                    subgroup_measurements.append(measurements[ap])
                except KeyError:
                    include_point = False
            if include_point:
                position = self.gradient_descent.train(subgroup_measurements, {"x": 0, "y": 0, "z": 0})
                all_pos.append(position)
                #ax.scatter(position['x'], position['y'], s=20, c=self.colors[subgroup], marker="x")
                all_gt = [x.ground_truth for x in subgroup_measurements]
                for gt in all_gt:
                    ax.scatter(gt['x'], gt['y'], s=20, c='b', marker="o")

        distanceAnalyzer = DistanceAnalyzer(all_pos)
        point_to_add = distanceAnalyzer.get_min_distance_point()
        smoothed_current_point = smooth_current_point(all_pos_distance_analyzer,point_to_add, particle_filter)
        all_pos_distance_analyzer.append(point_to_add)
        ax.scatter(smoothed_current_point['x'], smoothed_current_point['y'], s=20, c='purple', marker="x")
        ax.scatter(point_to_add['x'], point_to_add['y'], s=20, c='green', marker="x")
#----------------------------------------------------------------------------------------------------------------
        clusterAnalyzer = ClusterAnalyzer(all_pos)
        point_to_add = clusterAnalyzer.get_mean_of_largest_cluster()
        point_to_add={'x':point_to_add[0],'y':point_to_add[1],'z':point_to_add[2]}
        smoothed_current_point = smooth_current_point(all_pos_cluster_analyzer,point_to_add, particle_filter)
        all_pos_cluster_analyzer.append(point_to_add)
        ax.scatter(point_to_add['x'], point_to_add['y'], s=20, c='blue', marker="x")
        ax.scatter(smoothed_current_point['x'], smoothed_current_point['y'], s=20, c='black', marker="x")

        self.index += 1
        self.canvas.draw()
        filename = "plots_with_tracer/plot_{}.png".format(self.index)
        self.fig.savefig(filename)

if __name__ == "__main__":
    root = tk.Tk()

    root.title("WiFi path 6 of 4")
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    controls_frame = tk.Frame(root)
    controls_frame.pack(side=tk.BOTTOM)
    entry_exp = tk.Entry(controls_frame)
    entry_exp.insert(0, "EXP_56")
    entry_exp.pack(side=tk.LEFT)
    plotter = WiFiPathPlotter(root, fig, canvas, controls_frame, entry_exp)
    button = tk.Button(controls_frame, text="Plot", command=plotter.plot)
    button.pack(side=tk.LEFT)
    root.mainloop()

