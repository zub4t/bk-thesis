import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'..','classes'))
sys.path.insert(0,os.path.join(cwd,'..','commons'))
import unittest
from Util import calculate_distance

class TestUtil(unittest.TestCase):
    def test_calculate_distance(self):
        loc1 = {'x': 0, 'y': 0, 'z': 0}
        loc2 = {'x': 1, 'y': 1, 'z': 1}
        expected_distance = 1.7320508075688772
        actual_distance = calculate_distance(loc1, loc2)
        message = f"Expected distance {expected_distance}, but got {actual_distance}"
        self.assertAlmostEqual(actual_distance, expected_distance, places=4)

if __name__ == '__main__':
    unittest.main()

