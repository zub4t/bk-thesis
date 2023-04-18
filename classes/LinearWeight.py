
import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'interfaces'))
from IweightFunction import IweightFunction

class Func(IweightFunction):
    def perform(self, rssi: float) -> float:
        # Customize the slope (m) and y-intercept (b) as needed
        m = 1
        b = 100
        
        # Calculate the weight using a linear function
        weight = m * rssi + b

        return weight

