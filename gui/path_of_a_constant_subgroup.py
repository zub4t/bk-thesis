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
import Util
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
        self.bias = lambda x: x / 1.16 - 0.63

    def process(self, subgroup_size, exp):
        measurements_dict, mobile_location_dict = Util.read_json_file("../JSON/file.json", "802.11mc")
        filtered_dict = {
            k: [obj for obj in v if obj.exp == exp] for k, v in measurements_dict.items() if any(obj.exp == exp for obj in v)
        }
        sampled_list = random.sample(filtered_dict.keys(), 6)
        #sampled_list = ['5A0A','111F','D018','D713','120F','868C']
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
        ax.scatter(x, y, c="r", marker="o")
        measurement_buckets = Util.bucket_measurements(self.filtered_dict, 1000)
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
                ax.scatter(position['x'], position['y'], s=20, c=self.colors[subgroup], marker="x")
                all_gt = [x.ground_truth for x in subgroup_measurements]
                for gt in all_gt:
                    ax.scatter(gt['x'], gt['y'], s=20, c='b', marker="o")

        self.index += 1
        self.canvas.draw()
        filename = "plots_comparison/plot_{}.png".format(self.index)
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
    entry_exp.insert(0, "EXP_73")
    entry_exp.pack(side=tk.LEFT)
    plotter = WiFiPathPlotter(root, fig, canvas, controls_frame, entry_exp)
    button = tk.Button(controls_frame, text="Plot", command=plotter.plot)
    button.pack(side=tk.LEFT)
    root.mainloop()

