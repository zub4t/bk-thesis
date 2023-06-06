from scipy.optimize import minimize
import numpy as np

# Define the cost function
def cost_func(x, points):
    total_distance = 0
    for point in points:
        total_distance += np.sqrt((x[0] - point['x'])**2 + (x[1] - point['y'])**2 + (x[2] - point['z'])**2)
    return total_distance

# Define your points
points = [{'x': 1, 'y': 2,'z':3}, {'x': 2, 'y': 3,'z':3}, {'x': 3, 'y': 4,'z':3}]

# Initial guess for the 'x' and 'y' coordinates of the center
x0 = np.array([0, 0, 0])

# Call the minimize function
res = minimize(cost_func, x0, args=(points), method='Nelder-Mead')

# Print the result
print(res.x)

