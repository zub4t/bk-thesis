import sys
import os
cwd = os.getcwd()
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(cwd,'classes'))
sys.path.insert(0,os.path.join(cwd,'commons'))
from Measurement import Measurement
from Synthetic import Synthetic
from Util import *
from GradientDescent import GradientDescent
from GradientDescentFixedZ import GradientDescentFixedZ
from APLocation import APLocation
import matplotlib.animation as animation
ax = None
# Example usage
if __name__ == "__main__":

    print("Please choose a data option:")
    print("1. Use Synthetic data")
    print("2. Use 802.11mc data")
    print("3. Use Ultra Wide Band data")

    data_option = input("Enter the number of your choice: ")

    if data_option == "1":
        data_source = "Synthetic data"
    elif data_option == "2":
        data_source = "802.11mc data"
    elif data_option == "3":
        data_source = "Ultra Wide Band data"
    else:
        print("Invalid choice, please try again.")
        exit()

    exp_target = input("Enter the name of the experiment you want to visualize: ")
    print("You chose to use {} and visualize the experiment {}".format(data_source, exp_target))
    l=None
    d=None
    update_func= None
    fast=False
    if(data_option == "1"):
            num_points = int( 60 / 0.3)  # Assuming 4 minutes of path with a 0.3 seconds interval between points
            side_length = 8
            synthetic = Synthetic()
            synthetic.square_path(num_points,side_length)
            m = synthetic.generate_synthetic_data(12)
            update_func= generate_person_path(20,Synthetic.real_person_path)
            d = group_measurements_by_bssid(m)
            l = generate_subgroups(4, arr=list(d.keys()) )
    elif(data_option=="2"):
            m = Measurement.read_json_file('/home/marco/Documents/site-thesis/file.json',exp_target,'802.11mc')
            real_person_path=interpolate_points(Measurement.points_exp, 20)
            update_func = generate_person_path(20,real_person_path)
            print(m)
            d = group_measurements_by_bssid(m)
            l = generate_subgroups(4, arr=list(d.keys()) )
    elif(data_option=="3"):     
            m = Measurement.read_json_file('/home/marco/Documents/site-thesis/file.json',exp_target,'uwb')
            real_person_path=interpolate_points(Measurement.points_exp, 20)
            update_func = generate_person_path(20,real_person_path)
            d = group_measurements_by_bssid(m)
            l = generate_subgroups(4, arr=list(d.keys()) )
            fast=True

    gd = GradientDescentFixedZ(learning_rate=0.01, max_iterations=1000, tolerance=1e-5)
    timestamp_list = read_timestamps(f'/home/marco/Documents/raw_802.11_new/CHECKPOINT_{exp_target}')
    points_list = generate_intermediate_points(Measurement.points_exp)
    
    cc = (len(d[list(d.keys())[1]]))
    plt.show(block=False)
    for j in range(0,cc):
        if(fast):
            j*=20
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
            #pos['z'] = 1.6
            pos_dict[group] = pos
            error = calculate_distance(pos, measurements[1].responder_location)
            error_dict[group] = error
            print(f'{group} -- {j} --- {error}')
        min_key = min(error_dict, key=lambda k: error_dict[k])
        min_value = error_dict[min_key]
        pos = pos_dict[min_key]
        update_func([pos], color='b')
        plt.show(block=False)

        


