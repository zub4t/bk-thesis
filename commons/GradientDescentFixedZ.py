import sys
import os

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "..", "classes"))
from Util import calculate_distance


class GradientDescent:
    def __init__(self, learning_rate=0.1, max_iterations=1000000, tolerance=1e-5):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def gradient(self, measurements, target):
        grad = {"x": 0, "y": 0}
        epsilon = 1e-10
        for m in measurements:
            dist = calculate_distance(target, m.ap_location)
            if dist != None:
                dist = max(dist, epsilon)  # Avoid division by zero
                error = m.distance - dist
                grad["x"] += (error / dist) * (target["x"] - m.ap_location["x"])
                grad["y"] += (error / dist) * (target["y"] - m.ap_location["y"])
        return grad

    def cost_function(self, measurements, target):
        cost = 0
        for m in measurements:
            dist = calculate_distance(target, m.ap_location)
            error = 1000000
            if dist != None:
                error = m.distance - dist
            cost += error**2
        return cost

    def train(self, measurements, initial_guess):
        target = initial_guess.copy()
        target["z"] = 1.70  # Set the fixed z-axis value

        prev_cost = self.cost_function(measurements, target)
        improvement = float("inf")
        for i in range(self.max_iterations):
            grad = self.gradient(measurements, target)
            target["x"] += self.learning_rate * grad["x"]
            target["y"] += self.learning_rate * grad["y"]

            current_cost = self.cost_function(measurements, target)
            improvement = abs(current_cost - prev_cost)
            prev_cost = current_cost

            if improvement < self.tolerance:
                break

        return target
