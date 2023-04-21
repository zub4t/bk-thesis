import sys
import os
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'classes'))
sys.path.insert(0,os.path.join(cwd,'commons'))
from LinearWeight import Func  
from Measurement import Measurement,Synthetic 
from Util import group_measurements_by_bssid,generate_subgroups,measurements_to_location,calculate_distance
from GradientDescent import GradientDescent
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
    num_points = int( 60 / 0.3)  # Assuming 4 minutes of path with a 0.3 seconds interval between points
    side_length = 2
    s = Synthetic()
    s.square_path(num_points, side_length)
    m = s.generate_synthetic_data(12,total_time=60)
    d = group_measurements_by_bssid(m)
    l = generate_subgroups(8)
    z = int((4*60)//0.3)
    gd = GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)
    #print(d['ap_1'])
    for i, group in enumerate(l):
        for j in range(0,len(d['ap_1'])):
            measurements=[]
            for ap in group:
                measurements.append(d[ap][j])
            #pos = measurements_to_location(measurements)
            pos = gd.train(measurements, {'x':0,'y':0,'z':0})
            print(f'predict : {pos}  real {measurements[0].responder_location}')
            print(j)
            #print(measurements[0].timestamp)
