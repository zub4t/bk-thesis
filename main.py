# TODO: Check if all measurements in the line 134 has the same timestamp
import threading
import math
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Local modules
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, "classes"))
sys.path.insert(0, os.path.join(cwd, "commons"))
from APLocation import APLocation
from GradientDescent import GradientDescent
from GradientDescentFixedZ import GradientDescentFixedZ
from Measurement import Measurement
from Synthetic import Synthetic
from Util import (
    generate_intermediate_points,
    generate_person_path,
    generate_subgroups,
    group_measurements_by_bssid,
    interpolate_from_timestamp_to_location,
    interpolate_points,
    mean_location_1,
    read_timestamps,
    select_subset,
    mean_error,
    calculate_distance,
    append_to_file
)


# Define constants and settings
SMOOTHING_FACTOR_DEGREES = 30
USER_SPEED = 0.5


# Example usage
if __name__ == "__main__":
    print("Please choose a data option:")
    print("1. Use Synthetic data")
    print("2. Use 802.11mc data")
    print("3. Use Ultra Wide Band data")
    initial_group_size = 12
    data_option = input("Enter the number of your choice: ")
    experiment_target = None
    if data_option == "1":
        data_source = "Synthetic data"
    elif data_option == "2":
        data_source = "802.11mc data"
        experiment_target = input(
            "Enter the name of the experiment you want to visualize: "
        )
        print(
            f"You chose to use {data_source} and visualize the experiment {experiment_target}."
        )
    elif data_option == "3":
        data_source = "Ultra Wide Band data"
        experiment_target = input(
            "Enter the name of the experiment you want to visualize: "
        )
        print(
            f"You chose to use {data_source} and visualize the experiment {experiment_target}."
        )
    else:
        print("Invalid choice, please try again.")
        exit()

    # Load data and initialize variables
    measurements_dict = None
    subgroup_list = None
    update_func = None
    bias_func = lambda x: x
    fast = False
    if data_option == "1":
        num_points = int(
            60 / 0.3
        )  # Assuming 4 minutes of path with a 0.3 seconds interval between points
        side_length = 8
        synthetic = Synthetic()
        synthetic.square_path(num_points, side_length)
        measurements = synthetic.generate_synthetic_data(12)
        update_func = generate_person_path(20, Synthetic.real_person_path)
        measurements_dict = group_measurements_by_bssid(measurements)
    elif data_option == "2":
        #bias_func = lambda x: (x / 1.16) - 0.63
        measurements = Measurement.read_json_file(
            "./file.json", experiment_target, "802.11mc"
        )
        real_person_path = interpolate_points(Measurement.points_exp, 20)
        points_list = generate_intermediate_points(Measurement.points_exp)
        update_func = generate_person_path(20, real_person_path)
        measurements_dict = group_measurements_by_bssid(measurements)
        initial_group_size = 6
    elif data_option == "3":
        measurements = Measurement.read_json_file(
            "./file.json", experiment_target, "uwb"
        )
        real_person_path = interpolate_points(Measurement.points_exp, 20)
        update_func = generate_person_path(20, real_person_path)
        measurements_dict = group_measurements_by_bssid(measurements)
        fast = True
        initial_group_size = 12

    # Initialize gradient descent algorithm and visualization variables
    gradient_descent = GradientDescentFixedZ(
        learning_rate=0.01, max_iterations=1000, tolerance=1e-5
    )
    timestamp_list = []
    if data_option == "2":
        timestamp_list = read_timestamps(
            f"./CHECKPOINTS/CHECKPOINT_{experiment_target}"
        )
        points_list = generate_intermediate_points(Measurement.points_exp)

    if data_option == "3":
        aux = experiment_target.split('_')
        timestamp_list = read_timestamps(
            f"./CHECKPOINTS/CHECKPOINT_EXP_{int(aux[1])+17}"
        )
        points_list = generate_intermediate_points(Measurement.points_exp)
    num_measurements = len(measurements_dict[list(measurements_dict.keys())[1]])
    plt.show(block=False)

    def main_func(group_size):
        prev_mean_loc_list = [{"x": 0.1, "y": 0.1, "z": 0.1}]
        prev_mean_loc = {"x": 0.1, "y": 0.1, "z": 0.1}
        prev_timestamp = 0
        subgroup_list = generate_subgroups(
            (group_size), arr=list(measurements_dict.keys())
        )
        list_error = []
        #print(list_error)
        # Perform gradient descent and update visualization for each measurement
        for j in range(0, num_measurements):
            if fast:
                j *= 20
            tuple_list = []
            current_timestamp = 0
            current_ground_truth = None
            for i, subgroup in enumerate(subgroup_list):
                measurements = []
                for ap in subgroup:
                    try:
                        measurement = measurements_dict[ap][j]
                        if data_option != "1":
                            measurement.ground_truth = (
                                interpolate_from_timestamp_to_location(
                                    points_list,
                                    timestamp_list,
                                    measurement.timestamp,
                                )
                            )
                        measurement.distance = bias_func(measurement.distance)
                        measurements.append(measurement)
                    except:
                        print("error")
                current_ground_truth = measurements[0].ground_truth
                current_timestamp = measurements[0].timestamp
                position = gradient_descent.train(
                    measurements, {"x": 0, "y": 0, "z": 0}
                )
                # position['z'] = 1.6
                # print(position)
                tuple_list.append((subgroup, position))
                #tuple_list.append((subgroup, current_ground_truth))
                print(f"{subgroup} -- {j}")
            subset = select_subset(tuple_list, 0.5)
            mean_loc = mean_location_1(
                subset, current_timestamp, prev_mean_loc, prev_timestamp, USER_SPEED
            )
            error = calculate_distance(mean_loc, current_ground_truth)
            list_error.append(error)
            prev_timestamp = current_timestamp
            prev_mean_loc_list.append(mean_loc)
            prev_mean_loc = mean_loc
            test = []
            for key in measurements_dict:
                    test.append(measurements_dict[key][j])

            position_test = gradient_descent.train(
                test, {"x": 0, "y": 0, "z": 0}
            )
            update_func([mean_loc], color="b") 
            update_func([position_test], color="purple")
            update_func([current_ground_truth], color="g")
            plt.show(block=False)

        append_to_file(f'group_size_{group_size}.txt',str(mean_error(list_error)))

    main_func(4)
    # def call_main_func(group_size):
    #     main_func(group_size)
    #
    # threads = []
    # for w in range(0, 1):
    #     t = threading.Thread(target=call_main_func, args=(initial_group_size - w,))
    #     threads.append(t)
    #     t.start()
    #
    # for t in threads:
    #     t.join()

