import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'classes'))
sys.path.insert(0,os.path.join(cwd,'ThirdParties/SGD'))
from LinearWeight import Func  
from SGD import SGD as sgd
def give_weight(devices: dict):
    
    """
    Adds a weight property to each object in the devices dictionary based on their rssi value.

    :param devices: Dictionary with keys as device names (strings) and values as objects with an 'rssi' property
    """
    func = Func()
    for device_name, device_obj in devices.items():
        device_obj['weight'] = func.perform(device_obj['rssi'])

# Example usage
if __name__ == "__main__":
    devices = {
        'device1': {'rssi': -30},
        'device2': {'rssi': -60},
        'device3': {'rssi': -90},
    }
    give_weight(devices)
    print(devices)

