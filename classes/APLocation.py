import json


class APLocation:
    def __init__(self, pos, bssid, id, x, y, z):
        self.POS = pos
        self.BSSID = bssid
        self.ID = id
        self.X = x
        self.Y = y
        self.Z = z

    def __repr__(self):
        return f"APLocation(POS={self.POS}, BSSID={self.BSSID}, ID={self.ID}, X={self.X}, Y={self.Y}, Z={self.Z})"

    @staticmethod
    def from_file_to_list(file_path):
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)

        ap_locations = []
        for item in json_data:
            ap_location = APLocation(
                pos=item["POS"],
                bssid=item["BSSID"],
                id=item["ID"],
                x=item["X"],
                y=item["Y"],
                z=item["Z"],
            )
            ap_locations.append(ap_location)

        return ap_locations
