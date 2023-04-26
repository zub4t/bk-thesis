import sys
import os

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "..", "classes"))
sys.path.insert(0, os.path.join(cwd, "..", "commons"))
import unittest
from Util import *


class TestUtil(unittest.TestCase):
    def test_calculate_distance(self):
        loc1 = {"x": 0, "y": 0, "z": 0}
        loc2 = {"x": 1, "y": 1, "z": 1}
        expected_distance = 1.7320508075688772
        actual_distance = calculate_distance(loc1, loc2)
        message = f"Expected distance {expected_distance}, but got {actual_distance}"
        self.assertAlmostEqual(actual_distance, expected_distance, places=4)

    def test_intermedium_point(self):
        points = [
            {"x": 1, "y": 0.80, "z": 1.60},
            {"x": 8, "y": 0.80, "z": 1.60},
            {"x": 8, "y": 4.80, "z": 1.60},
            {"x": 1, "y": 4.80, "z": 1.60},
            {"x": 1, "y": 1.80, "z": 1.60},
            {"x": 8, "y": 1.80, "z": 1.60},
            {"x": 8, "y": 3.80, "z": 1.60},
            {"x": 1, "y": 3.80, "z": 1.60},
            {"x": 1, "y": 2.80, "z": 1.60},
            {"x": 8, "y": 2.80, "z": 1.60},
        ]
        timestamp_list = read_timestamps(
            "/home/marco/Documents/raw_802.11_new/CHECKPOINT_EXP_73"
        )
        points_list = generate_intermediate_points(points)
        r = interpolate_from_timestamp_to_location(
            points_list, timestamp_list, 1681726341960
        )
        print("location generated", r)


if __name__ == "__main__":
    unittest.main()
