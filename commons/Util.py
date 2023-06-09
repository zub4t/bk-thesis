import math
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
from more_itertools import distinct_combinations
import sys
import os
import random
import matplotlib.colors as mcolors

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "..", "classes"))
from Measurement import Measurement

arr_ap = [
    "ap_1",
    "ap_2",
    "ap_3",
    "ap_4",
    "ap_5",
    "ap_6",
    "ap_7",
    "ap_8",
    "ap_9",
    "ap_10",
    "ap_11",
    "ap_12",
]


def calculate_distance(location_1, location_2):
    try:
        x1, y1, z1 = (
            round(location_1["x"], 4),
            round(location_1["y"], 4),
            round(location_1["z"], 4),
        )
        x2, y2, z2 = (
            round(location_2["x"], 4),
            round(location_2["y"], 4),
            round(location_2["z"], 4),
        )
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    except OverflowError:
        print("Numerical result out of range")
        return None
    return distance


def calculate_distance_2D(location_1, location_2):
    x1, y1 = location_1["x"], location_1["y"]
    x2, y2 = location_2["x"], location_2["y"]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def location_obj_func(target, measurements):
    target = {"x": target[0], "y": target[1], "z": target[2]}
    error = 0
    for m in measurements:
        dist = np.sqrt(
            np.sum(
                (
                    np.array(list(target.values()))
                    - np.array(list(m.responder_location.values()))
                )
                ** 2
            )
        )
        error += (m.distance - dist) ** 2
    return error


def location_gradient(target, measurements):
    grad = {"x": 0, "y": 0, "z": 0}
    target = {"x": target[0], "y": target[1], "z": target[2]}
    for m in measurements:
        dist = np.sqrt(
            np.sum(
                (
                    np.array(list(target.values()))
                    - np.array(list(m.responder_location.values()))
                )
                ** 2
            )
        )
        error = m.distance - dist
        grad["x"] += (error / dist) * (target["x"] - m.responder_location["x"])
        grad["y"] += (error / dist) * (target["y"] - m.responder_location["y"])
        grad["z"] += (error / dist) * (target["z"] - m.responder_location["z"])
    return np.array(list(grad.values()))


def measurements_to_location(measurements):
    initial_guess = np.array([0, 0, 0])
    # Replace with your initial guess for the location
    # Define the bounds for the user's location
    bounds = ((0, 10), (0, 6), (0, 4))
    # initial_guess= np.array([0,0,0])
    result = minimize(
        location_obj_func,
        initial_guess,
        args=(measurements,),
        method="L-BFGS-B",
        jac=location_gradient,
        options={"disp": True, "maxiter": 500},
    )

    optimal_location = {"x": result.x[0], "y": result.x[1], "z": result.x[2]}
    return optimal_location


def generate_subgroups(group_size, arr=arr_ap):
    if group_size > len(arr):
        raise ValueError(
            "Group size cannot be larger than the number of available APs."
        )
    return list(distinct_combinations(arr, group_size))


def group_measurements_by_bssid(measurements):
    grouped_measurements = {}
    for measurement in measurements:
        # print(measurement)
        bssid = measurement.bssid
        if bssid not in grouped_measurements:
            grouped_measurements[bssid] = []
        grouped_measurements[bssid].append(measurement)
    return grouped_measurements


def interpolate_points(points, steps):
    interpolated_points = []

    for i in range(len(points) - 1):
        start_point = np.array([points[i]["x"], points[i]["y"], points[i]["z"]])
        end_point = np.array(
            [points[i + 1]["x"], points[i + 1]["y"], points[i + 1]["z"]]
        )
        for t in range(steps):
            alpha = t / (steps - 1)
            interpolated_point = start_point * (1 - alpha) + end_point * alpha
            interpolated_points.append(
                {
                    "x": interpolated_point[0],
                    "y": interpolated_point[1],
                    "z": interpolated_point[2],
                }
            )

    return interpolated_points


def interpolate(p1, p2, distance_ratio):
    return {
        "x": p1["x"] + distance_ratio * (p2["x"] - p1["x"]),
        "y": p1["y"] + distance_ratio * (p2["y"] - p1["y"]),
        "z": p1["z"] + distance_ratio * (p2["z"] - p1["z"]),
    }


def generate_intermediate_points(points, step=1.0):
    intermediate_points = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        segment_distance = calculate_distance(p1, p2)
        num_steps = int(segment_distance // step)
        for j in range(1, num_steps):
            distance_ratio = j * step / segment_distance
            intermediate_point = interpolate(p1, p2, distance_ratio)
            intermediate_points.append(intermediate_point)
        intermediate_points.append(p2)
    return intermediate_points


def read_timestamps(file_path):
    timestamps = []
    with open(file_path, "r") as file:
        for line in file:
            timestamp = int(line.strip())
            timestamps.append(timestamp)
    return timestamps


def interpolate_from_location_to_timestamp(points, timestamps, location):
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        t1 = timestamps[i]
        t2 = timestamps[i + 1]

        d1 = calculate_distance(p1, location)
        d2 = calculate_distance(p2, location)
        total_distance = calculate_distance(p1, p2)

        if d1 + d2 <= total_distance + 1e-6:
            distance_ratio = d1 / (d1 + d2)
            interpolated_timestamp = t1 + distance_ratio * (t2 - t1)
            return int(interpolated_timestamp)

    # If the location is not found between given points, use interpolation based on the first two points and timestamps
    p1, p2 = points[0], points[1]
    t1, t2 = timestamps[0], timestamps[1]
    total_distance = calculate_distance(p1, p2)
    d1 = calculate_distance(p1, location)
    distance_ratio = d1 / total_distance
    interpolated_timestamp = t1 + distance_ratio * (t2 - t1)
    return int(interpolated_timestamp)


def interpolate_from_timestamp_to_location(points, timestamps, target_timestamp):
    if target_timestamp <= timestamps[0]:
        return points[0]

    if target_timestamp >= timestamps[-1]:
        return points[-1]

    for i in range(len(timestamps) - 1):
        t1 = timestamps[i]
        t2 = timestamps[i + 1]

        if t1 <= target_timestamp <= t2:
            p1 = points[i]
            p2 = points[i + 1]
            time_ratio = (target_timestamp - t1) / (t2 - t1)
            x = p1["x"] + time_ratio * (p2["x"] - p1["x"])
            y = p1["y"] + time_ratio * (p2["y"] - p1["y"])
            z = p1["z"] + time_ratio * (p2["z"] - p1["z"])
            return {"x": x, "y": y, "z": z}

    # If the target timestamp is not found between given timestamps, use interpolation based on the first two points and timestamps
    p1, p2 = points[0], points[1]
    t1, t2 = timestamps[0], timestamps[1]
    time_ratio = (target_timestamp - t1) / (t2 - t1)
    x = p1["x"] + time_ratio * (p2["x"] - p1["x"])
    y = p1["y"] + time_ratio * (p2["y"] - p1["y"])
    z = p1["z"] + time_ratio * (p2["z"] - p1["z"])
    return {"x": x, "y": y, "z": z}


def generate_person_path(real_person_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = [point["x"] for point in real_person_path]
    y = [point["y"] for point in real_person_path]
    z = [point["z"] for point in real_person_path]

    ax.scatter(x, y, z, c="r", marker="o")

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 6])
    ax.set_zlim([0, 4])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("Person's Path")

    def update(new_points, color="b"):
        for point in new_points:
            ax.scatter(point["x"], point["y"], point["z"], c=color, marker="x")
            plt.draw()
            plt.pause(0.001)

    return update


def mean_error(errors):
    """Calculate the mean error given a list of errors.

    Args:
        errors (list): A list of errors.

    Returns:
        float: The mean error.
    """
    n = len(errors)
    if n == 0:
        return 0
    total_error = sum(errors)
    mean_error = total_error / n
    return mean_error

def append_to_file(filename, text):
    try:
        # Try to open the file in "append" mode
        with open(filename, 'a') as f:
            f.write(text + '\n')
    except FileNotFoundError:
        # If the file doesn't exist, create it and write the text to it
        with open(filename, 'w') as f:
            f.write(text + '\n')



def read_json_file(file_path,technology):
    with open(file_path, "r") as f:
        data = json.load(f)
        measurement_list = []
        #for exp_l in data["mobileLocationMap"]:
        print(data["mobileLocationMap"])
        for measurement_data in data["comparisonData"]:
                if(technology=="802.11mc"):
                    if len(measurement_data["id"]) > 4:
                        measurement = Measurement(
                            measurement_data["timestamp"],
                            measurement_data["id"],
                            measurement_data["measurement"],
                            measurement_data["groundTruth"],
                            measurement_data["pos"],
                            "802.11mc",
                            measurement_data["exp"]
                        )
                        measurement_list.append(measurement)
                else:
                    if len(measurement_data["id"]) <= 4:
                        measurement = Measurement(
                            measurement_data["timestamp"],
                            measurement_data["id"],
                            measurement_data["measurement"],
                            measurement_data["groundTruth"],
                            measurement_data["pos"],
                            "uwb",
                            measurement_data["exp"]
                        )
                        measurement_list.append(measurement)

        measurements_dict = group_measurements_by_bssid(measurement_list)
        return measurements_dict,data["mobileLocationMap"]


def generate_color_dict_v1(some_set):
    color_dict = {}
    # Get a list of all available named colors in Matplotlib
    all_colors = list(mcolors.CSS4_COLORS.keys())
    # Filter out light colors based on the combined RGB value
    colors = [color for color in all_colors if sum(mcolors.to_rgb(color)) < 1.5]
    for i, element in enumerate(some_set):
        # Randomly select a color from the list of named colors
        color_name = random.choice(colors)
        # Get the RGB tuple corresponding to the named color
        color_rgb = mcolors.CSS4_COLORS[color_name]
        # Add the element and color to the dictionary
        color_dict[element] = color_rgb
    return color_dict

def generate_color_dict(some_set):
    color_dict = {}
    # get a list of all available named colors in Matplotlib
    colors = list(mcolors.CSS4_COLORS.keys())
    for element in some_set:
        # randomly select a color from the list of named colors
        color_name = random.choice(colors)
        # get the RGB tuple corresponding to the named color
        color_rgb = mcolors.CSS4_COLORS[color_name]
        # add the element and color to the dictionary
        color_dict[element] = color_rgb
    return color_dict


def min_sum_distances_points(points):
    min_sum = float('inf')
    min_point = None

    for i, p1 in enumerate(points):
        sum_dist = 0
        for j, p2 in enumerate(points):
            if i != j:
                dist = math.sqrt((p2['x'] - p1['x']) ** 2 + (p2['y'] - p1['y']) ** 2 + (p2['z'] - p1['z']) ** 2)
                sum_dist += dist
        if sum_dist < min_sum:
            min_sum = sum_dist
            min_point = p1
    return min_point 

def filter_measurements(dict_of_measurements):
    # Find the key with the smallest number of measurements
    smallest_key = min(dict_of_measurements, key=lambda x: len(dict_of_measurements[x]))

    # Find the timestamp of the first and last measurement in the smallest key list
    smallest_timestamp_start = dict_of_measurements[smallest_key][0].timestamp
    smallest_timestamp_end = dict_of_measurements[smallest_key][-1].timestamp

    # Filter the measurements in each list to have the same number of measurements and matching timestamps
    filtered_measurements = {}
    for key, measurements in dict_of_measurements.items():
        # Find the first measurement with a timestamp >= smallest_timestamp_start
        i = 0
        while i < len(measurements) and measurements[i].timestamp < smallest_timestamp_start:
            i += 1

        # Find the last measurement with a timestamp <= smallest_timestamp_end
        j = len(measurements) - 1
        while j >= 0 and measurements[j].timestamp > smallest_timestamp_end:
            j -= 1

        # Add the filtered measurements to the output dictionary
        filtered_measurements[key] = measurements[i:j+1]

    return filtered_measurements
def get_measurements(data, keys=None, time_window=100):
    # If no list of keys is provided, use all keys in the dictionary
    if keys is None:
        keys = data.keys()

    # First, get the earliest timestamp among all first measurements of the given keys
    base_timestamp = min(data[k][0].timestamp for k in keys)

    # For each key, find the first measurement that falls within the time window of the base timestamp
    result = {}
    for key in keys:
        measurements = data.get(key)
        if measurements is not None:
            for measurement in measurements:
                if abs(measurement.timestamp - base_timestamp) <= time_window:
                    result[key]=measurement
                    break  # Once we find a match, we don't need to check the rest of the measurements for this key

    return result

def calculate_mean_point(points):
    if len(points) == 0:
        return {'x': 0, 'y': 0, 'z': 0}

    sum_x, sum_y, sum_z = 0, 0, 0
    for point in points:
        sum_x += point['x']
        sum_y += point['y']
        sum_z += point['z']

    mean_x = sum_x / len(points)
    mean_y = sum_y / len(points)
    mean_z = sum_z / len(points)

    return {'x': mean_x, 'y': mean_y, 'z': mean_z}

def bucket_measurements(measurements_dict, window_ms):
    # Create a sorted list of tuples (key, first timestamp)
    timestamps = sorted((key, m_list[0].timestamp) for key, m_list in measurements_dict.items() if m_list)
    base_timestamp = timestamps[-1][1]  # The biggest timestamp in the first position

    while True:
        bucket = {}
        for key, measurements in measurements_dict.items():
            # Filter measurements that are within the window
            within_window = [m for m in measurements if base_timestamp <= m.timestamp < base_timestamp + window_ms]
            if within_window:
                # If more than one measurement falls into this window, take the average
                avg_distance = sum(m.distance for m in within_window) / len(within_window)
                avg_timestamp = sum(m.timestamp for m in within_window) // len(within_window)
                avg_measurement = Measurement(avg_timestamp,key,avg_distance,0, within_window[0].ap_location)
                bucket[key] = avg_measurement
                # Remove used measurements from the original list
                measurements_dict[key] = [m for m in measurements if m not in within_window]

        if not bucket:
            break  # No more data

        yield bucket  # Return the current bucket and pause execution

        base_timestamp += window_ms
def create_cdf_plot(data, filename):
    sorted_data = np.sort(data)
    cumulative_probabilities = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, cumulative_probabilities, marker='.')
    plt.title("Cumulative Distribution Function")
    plt.xlabel("Values")
    plt.ylabel("Cumulative Probability")
    plt.savefig(filename)
    plt.clf()

