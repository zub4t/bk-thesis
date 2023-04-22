import math
from scipy.optimize import minimize
import numpy as np
from more_itertools import distinct_combinations
arr_ap = ['ap_1', 'ap_2', 'ap_3', 'ap_4', 'ap_5','ap_6','ap_7','ap_8','ap_9','ap_10','ap_11','ap_12']
def calculate_distance(location_1, location_2):
    try:
        x1, y1, z1 = round(location_1['x'],4), round(location_1['y'],4), round(location_1['z'],4)
        x2, y2, z2 = round(location_2['x'],4), round(location_2['y'],4), round(location_2['z'],4)
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    except OverflowError:
        print("Numerical result out of range")
        return None 
    return distance
def calculate_distance_2D(location_1, location_2):
    x1, y1 = location_1['x'], location_1['y']
    x2, y2 = location_2['x'], location_2['y']
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
def location_obj_func(target, measurements):
    target = {'x': target[0], 'y': target[1], 'z': target[2]}
    error = 0
    for m in measurements:
        dist = np.sqrt(np.sum((np.array(list(target.values())) - np.array(list(m.responder_location.values())))**2))
        error += (m.distance - dist)**2
    return error

def location_gradient(target, measurements):
    grad = {'x': 0, 'y': 0, 'z': 0}
    target = {'x': target[0], 'y': target[1], 'z': target[2]}
    for m in measurements:
        dist = np.sqrt(np.sum((np.array(list(target.values())) - np.array(list(m.responder_location.values())))**2))
        error = m.distance - dist
        grad['x'] += (error / dist) * (target['x'] - m.responder_location['x'])
        grad['y'] += (error / dist) * (target['y'] - m.responder_location['y'])
        grad['z'] += (error / dist) * (target['z'] - m.responder_location['z'])
    return np.array(list(grad.values()))

def measurements_to_location(measurements):
        initial_guess = np.array([0,0,0])
        # Replace with your initial guess for the location
        # Define the bounds for the user's location
        bounds = ((0, 10), (0, 6), (0, 4))       
       # initial_guess= np.array([0,0,0])
        result = minimize(location_obj_func, initial_guess, args=(measurements,), method='L-BFGS-B', jac=location_gradient, options={'disp': True, 'maxiter': 500})

        optimal_location = {'x': result.x[0], 'y': result.x[1], 'z': result.x[2]}
        return optimal_location

def generate_subgroups(group_size):
    if group_size > len(arr_ap):
        raise ValueError("Group size cannot be larger than the number of available APs.")
    return list(distinct_combinations(arr_ap, group_size))

def group_measurements_by_bssid(measurements):
    grouped_measurements = {}
    for measurement in measurements:
        bssid = measurement.bssid
        if bssid not in grouped_measurements:
            grouped_measurements[bssid] = []
        grouped_measurements[bssid].append(measurement)
    return grouped_measurements




def interpolate_points(points, steps):
    interpolated_points = []

    for i in range(len(points) - 1):
        start_point = np.array([points[i]['x'], points[i]['y'], points[i]['z']])
        end_point = np.array([points[i+1]['x'], points[i+1]['y'], points[i+1]['z']])
        for t in range(steps):
            alpha = t / (steps - 1)
            interpolated_point = start_point * (1 - alpha) + end_point * alpha
            interpolated_points.append({'x': interpolated_point[0], 'y': interpolated_point[1], 'z': interpolated_point[2]})

    return interpolated_points



