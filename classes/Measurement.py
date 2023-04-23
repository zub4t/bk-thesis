import sys
import os
import json
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd,'..','commons'))
class Measurement:
    points_exp = [
                {'x': 1, 'y': 0.80, 'z': 1.60},
                {'x': 8, 'y': 0.80, 'z': 1.60},
                {'x': 8, 'y': 4.80, 'z': 1.60},
                {'x': 1, 'y': 4.80, 'z': 1.60},
                {'x': 1, 'y': 1.80, 'z': 1.60},
                {'x': 8, 'y': 1.80, 'z': 1.60},
                {'x': 8, 'y': 3.80, 'z': 1.60},
                {'x': 1, 'y': 3.80, 'z': 1.60},
                {'x': 1, 'y': 2.80, 'z': 1.60},
                {'x': 8, 'y': 2.80, 'z': 1.60},
                ]
    real_person_path =[]
    def __init__(self, timestamp, bssid,distance, ground_truth, ap_location):
        self.timestamp = timestamp
        self.bssid = bssid
        self.distance = distance
        self.ground_truth = ground_truth
        self.ap_location = ap_location
    def __repr__(self):
        return (f"Distance: {self.distance:.2f}\n")
    
    @staticmethod
    def read_json_file(file_path,exp_target,technology):
        with open(file_path, 'r') as f:
            data = json.load(f)
            measurement_list = []
            for measurement_data in data['comparisonData']:
                if measurement_data['exp'] == exp_target :
                    if technology == '802.11mc': 
                        if len(measurement_data['id']) > 4: 
                            measurement = Measurement(
                                measurement_data['timestamp'],
                                measurement_data['id'],
                                measurement_data['measurement'],
                                measurement_data['groundTruth'],
                                measurement_data['pos']
                            )
                            measurement_list.append(measurement)
                    else:
                        if len(measurement_data['id']) == 4:
                            measurement = Measurement(
                                measurement_data['timestamp'],
                                measurement_data['id'],
                                measurement_data['measurement'],
                                measurement_data['groundTruth'],
                                measurement_data['pos']
                            )
                            measurement_list.append(measurement)
            return measurement_list
