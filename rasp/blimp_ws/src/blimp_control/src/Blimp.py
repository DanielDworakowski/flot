import re
from bluetooth.ble import DiscoveryService
from bluetooth.ble import GATTRequester

class Blimp:

    def __init__(self, mac="C4:C3:00:01:07:3F"):
        # Check for empty arg and correct MAC address format
        # Default MAC address is given otherwise
        if not re.match("[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac.lower()):
            print("Using default MAC: C4:C3:00:01:07:3F")
            self.mac = "C4:C3:00:01:07:3F"
        else:
            self.mac = mac

        self.service = DiscoveryService()
        self.devices = self.service.discover(2)
        self.requester = GATTRequester(self.mac, False)

    def find_blimp(self, blimp_name):
        self.devices = self.service.discover(2)
        for address, name in self.devices:
            if name == blimp_name:
                self.mac = address
                print(blimp_name + " found with MAC Address " + address)
                break

    def connect(self):
        try:
            self.requester.connect(True)
        except:
            print("Failed to connect; Make sure target is turned on and correct MAC address is provided")

    def disconnect(self):
        try:
            self.requester.disconnect(True)
        except:
            print("Failed to disconnect; try again")

    def is_connected(self):
        return self.requester.is_connected()

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for left propeller
    def left(self, value):

        if self.is_connected():
            if -32768 < value < 32768:
                if value < 0:
                    command = '{:04x}'.format(-1*int(value))
                else:
                    command = '{:04x}'.format(65535 - int(value))

                self.requester.write_by_handle(34, command.decode('hex'))
            else:
                print("Command value is must be integer between -32767 & 32767")
        else:
            print("First connect to target before commanding thrust")

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for right propeller
    def right(self, value):
        
        if self.is_connected():
            if -32768 < value < 32768:
                if value < 0:
                    command = '{:04x}'.format(-1*int(value))
                else:
                    command = '{:04x}'.format(65535 - int(value))

                self.requester.write_by_handle(36, command.decode('hex'))
            else:
                print("Command value is must be integer between -32767 & 32767")
        else:
            print("First connect to target before commanding thrust")

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for down propeller
    def down(self, value):

        if self.is_connected():
            if -32768 < value < 32768:
                if value < 0:
                    command = '{:04x}'.format(-1*int(value))
                else:
                    command = '{:04x}'.format(65535 - int(value))

                self.requester.write_by_handle(38, command.decode('hex'))
            else:
                print("Command value is must be integer between -32767 & 32767")
        else:
            print("First connect to target before commanding thrust")

    # Function to stop all actuators
    def stop(self):
        if self.is_connected():
                command = '{:04x}'.format(65535)
                self.requester.write_by_handle(34, command.decode('hex'))
                self.requester.write_by_handle(36, command.decode('hex'))
                self.requester.write_by_handle(38, command.decode('hex'))
        else:
            print("Command failed; not connected to target")

