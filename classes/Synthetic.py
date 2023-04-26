import random
import sys
import os

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "..", "commons"))
from Measurement import Measurement
from Util import calculate_distance


class Synthetic:
    real_person_path = []

    def __init__(self):
        pass

    def generate_synthetic_data(
        self, num_aps, ap_locations={}, room_size=(10, 6, 4), time_interval=0.3
    ):
        # Generate random AP locations
        if len(ap_locations) == 0:
            for i in range(1, num_aps + 1):
                ap_locations[f"ap_{i}"] = {
                    "x": random.uniform(0, room_size[0]),
                    "y": random.uniform(0, room_size[1]),
                    "z": random.uniform(0, room_size[2]),
                }

        # Generate measurements
        measurements = []
        for i, person_location in enumerate(Synthetic.real_person_path):
            timestamp = i * time_interval
            for ap_name, ap_location in ap_locations.items():
                distance = calculate_distance(person_location, ap_location)
                distance_plus_noise = distance + random.uniform(-1.0, 1.0)
                measurement = Measurement(
                    timestamp,
                    ap_name,
                    distance_plus_noise,
                    person_location,
                    ap_location,
                )
                measurements.append(measurement)

        return measurements

    def generate_random_path(self, num_points, room_size=(10, 6, 4), max_distance=0.2):
        for _ in range(num_points):
            if (
                not Synthetic.real_person_path
            ):  # If person_path is empty, generate the first point
                point = {
                    "x": random.uniform(0, room_size[0]),
                    "y": random.uniform(0, room_size[1]),
                    "z": random.uniform(0, room_size[2]),
                }
                Synthetic.real_person_path.append(point)
            else:
                while True:
                    candidate_point = {
                        "x": random.uniform(0, room_size[0]),
                        "y": random.uniform(0, room_size[1]),
                        "z": random.uniform(0, room_size[2]),
                    }
                    if (
                        calculate_distance(
                            Synthetic.real_person_path[-1], candidate_point
                        )
                        <= max_distance
                    ):
                        Synthetic.real_person_path.append(candidate_point)
                        break
        return Synthetic.real_person_path

    def square_path(self, num_points, side_length, room_size=(10, 6, 4), height=1.6):
        assert (
            side_length * 4 <= num_points
        ), "Number of points must be at least 4 times the side length"

        def next_square_point(current_point, side_length, direction):
            new_point = current_point.copy()
            if direction == 0:  # Move in positive x direction
                new_point["x"] += side_length
            elif direction == 1:  # Move in positive y direction
                new_point["y"] += side_length
            elif direction == 2:  # Move in negative x direction
                new_point["x"] -= side_length
            else:  # Move in negative y direction
                new_point["y"] -= side_length
            return new_point

        start_point = {
            "x": room_size[0] / 2 - side_length / 2,
            "y": room_size[1] / 2 - side_length / 2,
            "z": height,
        }
        Synthetic.real_person_path.append(start_point)

        points_per_side = num_points // 4
        for i in range(1, num_points):
            direction = (i // points_per_side) % 4
            new_point = next_square_point(
                Synthetic.real_person_path[-1], side_length / points_per_side, direction
            )
            Synthetic.real_person_path.append(new_point)
