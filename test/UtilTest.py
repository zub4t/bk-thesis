import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'..','classes'))
sys.path.insert(0,os.path.join(cwd,'..','commons'))
import unittest
from Util import calculate_distance,generate_subgroups

class TestUtil(unittest.TestCase):
    def test_calculate_distance(self):
        loc1 = {'x': 0, 'y': 0, 'z': 0}
        loc2 = {'x': 1, 'y': 1, 'z': 1}
        expected_distance = 1.7320508075688772
        actual_distance = calculate_distance(loc1, loc2)
        message = f"Expected distance {expected_distance}, but got {actual_distance}"
        self.assertAlmostEqual(actual_distance, expected_distance, places=4)


    def test_generate_subgroups(self):
        # Test case 1: group size is 3
        expected_subgroups = [('ap_1', 'ap_2', 'ap_3'),
                              ('ap_1', 'ap_2', 'ap_4'),
                              ('ap_1', 'ap_2', 'ap_5'),
                              ('ap_1', 'ap_3', 'ap_4'),
                              ('ap_1', 'ap_3', 'ap_5'),
                              ('ap_1', 'ap_4', 'ap_5'),
                              ('ap_2', 'ap_3', 'ap_4'),
                              ('ap_2', 'ap_3', 'ap_5'),
                              ('ap_2', 'ap_4', 'ap_5'),
                              ('ap_3', 'ap_4', 'ap_5')]
        self.assertEqual(generate_subgroups(3), expected_subgroups)
if __name__ == '__main__':
    unittest.main()

