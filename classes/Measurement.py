class Measurement:
    def __init__(self, timestamp, bssid, rssi, distance, std_dev, responder_location):
        self.timestamp = timestamp
        self.bssid = bssid
        self.rssi = rssi
        self.distance = distance
        self.std_dev = std_dev
        self.responder_location = responder_location

