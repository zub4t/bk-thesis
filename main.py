import sys
import os
import threading
cwd = os.getcwd()
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(cwd,'classes'))
sys.path.insert(0,os.path.join(cwd,'commons'))
from LinearWeight import Func  
from Measurement import Measurement,Synthetic 
from Util import *
from GradientDescent import GradientDescent
from APLocation import APLocation
import matplotlib.animation as animation
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
    m = []
    m += Measurement.from_folder_to_list_uwb('/home/marco/Documents/raw_new_uwb/UWB_0/parsed/EXP_56',ap_location)
    m += Measurement.from_folder_to_list_uwb('/home/marco/Documents/raw_new_uwb/UWB_1/parsed/EXP_56',ap_location)
    m += Measurement.from_folder_to_list_uwb('/home/marco/Documents/raw_new_uwb/UWB_2/parsed/EXP_56',ap_location)
    m[0].generate_exp_path(20)
    update_func = m[0].generate_person_path()
    d = group_measurements_by_bssid(m)
    l = generate_subgroups(12, arr=list(d.keys()) )
    gd = GradientDescent(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)
    

    timestamp_list = read_timestamps('/home/marco/Documents/raw_802.11_new/CHECKPOINT_EXP_73')
    points_list = generate_intermediate_points(m[0].points_exp)
    
    # for xx in m:
    #     pos = interpolate_from_timestamp_to_location(points_list,timestamp_list,xx.timestamp)
    #     ax.scatter(pos['x'],pos['y'],pos['z'],c='g', marker='x')
    # plt.show()
    chosen_points = []
    cc = (len(d[list(d.keys())[1]]))
    plt.show(block=False)
    for j in range(0,cc): 
        error_dict = {}
        pos_dict = {}
        for i, group in enumerate(l):
            measurements=[]
            for ap in group:
                try:
                    aux = d[ap][j]
                    aux.responder_location= interpolate_from_timestamp_to_location(points_list,timestamp_list,aux.timestamp)
                    measurements.append(aux)
                except:
                    print('error')
                
            pos = gd.train(measurements, {'x':0,'y':0,'z':0})
            # to verify the path is correct
            #pos_dict[group] = measurements[0].responder_location
            pos['z'] = 1.6
            pos_dict[group] = pos
            error = calculate_distance(pos, measurements[1].responder_location)
            error_dict[group] = error
            #print(f'predict : {pos}  real {measurements[0].responder_location}')
            print(f'{group} -- {j} --- {error}')
            #print(measurements[0].timestamp)
        min_key = min(error_dict, key=lambda k: error_dict[k])
        min_value = error_dict[min_key]
        pos = pos_dict[min_key]
        chosen_points.append(pos)
        #print(j,min_key, min_value,pos)
        # Add new points to the plot
        #ax.scatter(pos['x'],pos['y'],pos['z'],c='g', marker='x')
    update_func(chosen_points, color='b')

    plt.show()
        


