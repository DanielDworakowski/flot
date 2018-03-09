import re
import threading
from bluepy import *
import codecs

class Prop(object):

    def __init__(self, mac="C4:C3:00:01:07:3F"):
        # Check for empty arg and correct MAC address format
        # Default MAC address is given otherwise
        if not re.match("[0-9a-f]{2}([-:])[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac.lower()):
            print("Using default MAC: C4:C3:00:01:07:3F")
            self.mac = "C4:C3:00:01:07:3F"
        else:
            self.mac = mac
        self.shutdown = threading.Event()
        self.watchdog_thread = threading.Thread(target=self._watchdog, name="watchdog")
        self.requester = btle.Peripheral(self.mac)

    def is_connected(self):
        try:
            self.requester.readCharacteristic(1)
            return True
        except:
            return False

    def _watchdog(self):
        failCnt = 0
        while True:
            if self.shutdown.wait(timeout=5):
                failCnt = 0
                return
            if not self.is_connected():
                failCnt += 1
                print('Failed to connect. On attempt ', failCnt)

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for left propeller
    def left(self, value):
        if -32768 < value < 32768:
            if value < 0:
                command = '{:04x}'.format(-1*int(value))
            else:
                command = '{:04x}'.format(65535 - int(value))

            self.requester.writeCharacteristic(34, command.decode('hex'))
        else:
            print("Left command value must be integer between -32767 & 32767; received {}".format(value))

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for right propeller
    def right(self, value):
        if -32768 < value < 32768:
            if value < 0:
                command = '{:04x}'.format(-1*int(value))
            else:
                command = '{:04x}'.format(65535 - int(value))

            self.requester.writeCharacteristic(36, command.decode('hex'))
        else:
            print("Right command value must be integer between -32767 & 32767; received {}".format(value))

    # Enter value between -32767 to 32767
    # Negative value commands backward thrust, and vice versa with positive value, for down propeller
    def down(self, value):
        if -32768 < value < 32768:
            if value < 0:
                command = '{:04x}'.format(-1*int(value))
            else:
                command = '{:04x}'.format(65535 - int(value))

            self.requester.writeCharacteristic(38, command.decode('hex'))
        else:
            print("Down command value must be integer between -32767 & 32767; received {}".format(value))

    # Function to stop all actuators
    def stop(self):
        command = '{:04x}'.format(65535)
        self.requester.writeCharacteristic(34, command.decode('hex'))
        self.requester.writeCharacteristic(36, command.decode('hex'))
        self.requester.writeCharacteristic(38, command.decode('hex'))

    # Get battery level of propellers in integer values
    def batteryLevel(self):
       byte = self.requester.readCharacteristic(14)
       return int(codecs.encode(byte, 'hex'), 16)
