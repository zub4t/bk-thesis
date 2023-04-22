import sys
import os
import threading
cwd = os.getcwd()
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(cwd,'classes'))
sys.path.insert(0,os.path.join(cwd,'commons'))
from LinearWeight import Func  
from Measurement import Measurement,Synthetic 
from Util import group_measurements_by_bssid,generate_subgroups,measurements_to_location,calculate_distance
from GradientDescent import GradientDescent
from APLocation import APLocation
ax = None
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
    side_length = 8
    ap_location = APLocation.from_file_to_list('/home/marco/Documents/NEW/AP_location.json')
    m = Measurement.from_folder_to_list('/home/marco/Documents/raw_802.11_new/parsed/EXP_73',ap_location)
    d = group_measurements_by_bssid(m)
    l = generate_subgroups(12)
    """
    s = Synthetic()
    #s.square_path(num_points, side_length)
    #s.generate_random_path(num_points,max_distance=4)
    s.generate_exp_path(20)
    m = s.generate_synthetic_data(12,total_time=60)
    z = int((4*60)//0.3)
    gd = GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)
    j = 0
    ax = s.generate_person_path()
    plt.show()
    for j in range(0,len(d['ap_1'])): 
        error_dict = {}
        pos_dict = {}
        for i, group in enumerate(l):
            measurements=[]
            for ap in group:
                measurements.append(d[ap][j])
                #pos = measurements_to_location(measurements)
            pos = gd.train(measurements, {'x':0,'y':0,'z':0})
            pos_dict[group] = pos
            error = calculate_distance(pos, measurements[1].responder_location)
            error_dict[group] = error
            #print(f'predict : {pos}  real {measurements[0].responder_location}')
            #print(f'{group} -- {j} --- {error}')
            #print(measurements[0].timestamp)
        min_key = min(error_dict, key=lambda k: error_dict[k])
        min_value = error_dict[min_key]
        pos = pos_dict[min_key]
        print(j,min_key, min_value,pos)
        # Add new points to the plot
        ax.scatter(pos['x'],pos['y'],pos['z'],c='g', marker='x')
    plt.show()
    """
        


