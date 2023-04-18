import math
import numpy as np

def calculate_distance(location_1, location_2):
    x1, y1, z1 = location_1['x'], location_1['y'], location_1['z']
    x2, y2, z2 = location_2['x'], location_2['y'], location_2['z']
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
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

