import random
from collections import namedtuple
import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'..','commons'))
from Util import calculate_distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Measurement:
    def __init__(self, timestamp, bssid, rssi, distance, std_dev, responder_location, ap_location):
        self.timestamp = timestamp
        self.bssid = bssid
        self.rssi = rssi
        self.distance = distance
        self.std_dev = std_dev
        self.responder_location = responder_location
        self.ap_location = ap_location
    def __repr__(self):
        return (f"Timestamp: {self.timestamp:.1f}, BSSID: {self.bssid}, RSSI: {self.rssi:.2f}, Distance: {self.distance:.2f}, Std Dev: {self.std_dev:.2f}, Responder Location: {self.responder_location}, AP Location: {self.ap_location}")
class Synthetic:
    def __init__(self):
        self.person_path = []

    def generate_synthetic_data(self,num_aps, room_size=(10, 6, 4), rssi_range=(-100, -30), std_dev_range=(1, 5), total_time=4*60, time_interval=0.3):
        # Generate random AP locations
        ap_locations = {}
        for i in range(1, num_aps + 1):
            ap_locations[f"ap_{i}"] = {'x': random.uniform(0, room_size[0]), 'y': random.uniform(0, room_size[1]), 'z': random.uniform(0, room_size[2])}

        # Generate measurements
        measurements = []
        for i, person_location in enumerate(self.person_path):
            timestamp = i * time_interval
            for ap_name, ap_location in ap_locations.items():
                rssi = random.uniform(*rssi_range)
                distance = calculate_distance(person_location, ap_location)
                distance_plus_noise = distance + random.uniform(-1.0,1.0) 
                std_dev = random.uniform(*std_dev_range)
                measurement = Measurement(timestamp, ap_name, rssi, distance, std_dev, person_location, ap_location)
                measurements.append(measurement)

        return measurements

    def visualize_person_path(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = [point['x'] for point in self.person_path]
        y = [point['y'] for point in self.person_path]
        z = [point['z'] for point in self.person_path]

        ax.scatter(x, y, z, c='r', marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title("Person's Path")
        plt.show()



    def generate_random_path(self,num_points, room_size=(10, 6, 4), max_distance=0.2):
        for _ in range(num_points):
            if not self.person_path:  # If person_path is empty, generate the first point
                point = {'x': random.uniform(0, room_size[0]), 'y': random.uniform(0, room_size[1]), 'z': random.uniform(0, room_size[2])}
                self.person_path.append(point)
            else:
                while True:
                    candidate_point = {'x': random.uniform(0, room_size[0]), 'y': random.uniform(0, room_size[1]), 'z': random.uniform(0, room_size[2])}
                    if calculate_distance(self.person_path[-1], candidate_point) <= max_distance:
                        self.person_path.append(candidate_point)
                        break
        return self.person_path

    def square_path(self,num_points, side_length, room_size=(10, 6, 4), height=1.6):
        assert side_length * 4 <= num_points, "Number of points must be at least 4 times the side length"

        def next_square_point(current_point, side_length, direction):
            new_point = current_point.copy()
            if direction == 0:  # Move in positive x direction
                new_point['x'] += side_length
            elif direction == 1:  # Move in positive y direction
                new_point['y'] += side_length
            elif direction == 2:  # Move in negative x direction
                new_point['x'] -= side_length
            else:  # Move in negative y direction
                new_point['y'] -= side_length
            return new_point

        start_point = {'x': room_size[0] / 2 - side_length / 2, 'y': room_size[1] / 2 - side_length / 2, 'z': height}
        self.person_path.append(start_point)

        points_per_side = num_points // 4
        for i in range(1, num_points):
            direction = (i // points_per_side) % 4
            progress = i % points_per_side
            new_point = next_square_point(self.person_path[-1], side_length / points_per_side, direction)
            self.person_path.append(new_point)



