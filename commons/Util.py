import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
from more_itertools import distinct_combinations
import sys
import os

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
        print(measurement)
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


def generate_person_path(steps, real_person_path):
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


def select_subset(data, threshold):
    def is_close(point1, point2):
        return calculate_distance(point1, point2) < threshold

    subsets = []
    for element in data:
        ap_combination, location = element
        current_subset = [element]

        for other_element in data:
            if other_element != element:
                _, other_location = other_element
                if is_close(location, other_location):
                    current_subset.append(other_element)

        subsets.append(current_subset)

    # Find the largest subset
    largest_subset = max(subsets, key=len)
    return largest_subset


def calculate_raw_mean_location(subset):
    total_x = 0
    total_y = 0
    total_z = 0
    n = len(subset)

    for element in subset:
        _, location = element
        total_x += location["x"]
        total_y += location["y"]
        total_z += location["z"]

    return {"x": total_x / n, "y": total_y / n, "z": total_z / n}


def move_towards(raw_loc, prev_loc, ratio):
    return {
        "x": prev_loc["x"] + (raw_loc["x"] - prev_loc["x"]) * ratio,
        "y": prev_loc["y"] + (raw_loc["y"] - prev_loc["y"]) * ratio,
        "z": prev_loc["z"] + (raw_loc["z"] - prev_loc["z"]) * ratio,
    }


def mean_location_1(subset, current_ts, prev_mean_loc, prev_ts, speed):
    total_x = 0
    total_y = 0
    total_z = 0
    n = len(subset)

    for element in subset:
        _, location = element
        total_x += location["x"]
        total_y += location["y"]
        total_z += location["z"]

    raw_mean_x = total_x / n
    raw_mean_y = total_y / n
    raw_mean_z = total_z / n

    time_diff = current_ts - prev_ts
    max_distance = speed * time_diff
    distance = calculate_distance(
        prev_mean_loc, {"x": raw_mean_x, "y": raw_mean_y, "z": raw_mean_z}
    )

    if distance <= max_distance:
        mean_x = raw_mean_x
        mean_y = raw_mean_y
        mean_z = raw_mean_z
    else:
        ratio = max_distance / distance
        mean_x = prev_mean_loc["x"] + (raw_mean_x - prev_mean_loc["x"]) * ratio
        mean_y = prev_mean_loc["y"] + (raw_mean_y - prev_mean_loc["y"]) * ratio
        mean_z = prev_mean_loc["z"] + (raw_mean_z - prev_mean_loc["z"]) * ratio

    return {"x": mean_x, "y": mean_y, "z": mean_z}


def mean_location(subset, current_ts, prev_mean_locs, prev_ts, speed):
    raw_mean_loc = calculate_raw_mean_location(subset)

    time_diff = current_ts - prev_ts
    max_distance = speed * time_diff
    distance = calculate_distance(prev_mean_locs[-1], raw_mean_loc)

    if distance <= max_distance:
        if len(prev_mean_locs) > 0:
            weighted_mean_x = 0
            weighted_mean_y = 0
            weighted_mean_z = 0
            total_weight = 0

            for i, prev_mean_loc in enumerate(reversed(prev_mean_locs)):
                weight = 1 / (i + 1)  # More weight for recent locations
                total_weight += weight

                weighted_mean_x += prev_mean_loc["x"] * weight
                weighted_mean_y += prev_mean_loc["y"] * weight
                weighted_mean_z += prev_mean_loc["z"] * weight

            weighted_mean_x /= total_weight
            weighted_mean_y /= total_weight
            weighted_mean_z /= total_weight

            alpha = 0.5  # A tuning parameter between 0 and 1 to blend the raw mean location and the weighted moving average
            mean_x = alpha * raw_mean_loc["x"] + (1 - alpha) * weighted_mean_x
            mean_y = alpha * raw_mean_loc["y"] + (1 - alpha) * weighted_mean_y
            mean_z = alpha * raw_mean_loc["z"] + (1 - alpha) * weighted_mean_z

            mean_loc = {"x": mean_x, "y": mean_y, "z": mean_z}
        else:
            mean_loc = raw_mean_loc
    else:
        mean_loc = move_towards(
            raw_mean_loc, prev_mean_locs[-1], max_distance / distance
        )

    return mean_loc
