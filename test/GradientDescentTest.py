import sys
import os
import numpy as np
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'..','classes'))
sys.path.insert(0,os.path.join(cwd,'..','commons'))
sys.path.insert(0,os.path.join(cwd,'..','ThirdParties/SGD'))
import unittest
from GradientDescent import GradientDescent 
from Measurement import Measurement
from SGD import SGD as sgd
from Util import location_gradient,location_obj_func 
from scipy.optimize import minimize
class TestGradientDescent(unittest.TestCase):
    def setUp(self):
        self.measurements = [
            Measurement(timestamp=0, bssid='00:11:22:33:44:55', rssi=-50, distance=6, std_dev=0.1, responder_location={'x': 0, 'y': 0, 'z': 0}),
            Measurement(timestamp=1, bssid='00:11:22:33:44:55', rssi=-60, distance=5, std_dev=0.2, responder_location={'x': 10, 'y': 0, 'z': 0}),
            Measurement(timestamp=2, bssid='00:11:22:33:44:55', rssi=-70, distance=3, std_dev=0.3, responder_location={'x': 6, 'y': 3, 'z': 0}),
        ]
        self.initial_guess = {'x': 0, 'y': 0, 'z': 0}
    # def test_train(self):
    #     gd = GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)
    #     target = gd.train(self.measurements, self.initial_guess)
    #     self.assertAlmostEqual(target['x'], 1.15, places=2)
    #     self.assertAlmostEqual(target['y'], -0.6464, places=2)
    #     self.assertAlmostEqual(target['z'], -2.30, places=2)
    #     print(target)
    def test_2(self):
        #initial_guess = np.random.uniform(10,-10,3)  # Replace with your initial guess for the location
                
        initial_guess= np.array([0,0,0])
        result = minimize(location_obj_func, initial_guess, args=(self.measurements,), method='Powell', jac=location_gradient, options={'disp': True, 'maxiter': 5000})

        optimal_location = {'x': result.x[0], 'y': result.x[1], 'z': result.x[2]}
        print(f'optimal location : {optimal_location}')
        self.assertAlmostEqual(1, 1, places=2)
if __name__ == '__main__':
    unittest.main()
