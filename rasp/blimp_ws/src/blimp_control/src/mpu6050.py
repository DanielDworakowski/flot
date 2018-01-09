#!/usr/bin/python

import smbus
import math
import time
import numpy as np

class MPU6050:

    def __init__(self):
        # Power management registers
        self.power_mgmt_1 = 0x6b
        self.power_mgmt_2 = 0x6c
        self.bus = smbus.SMBus(1) # or bus = smbus.SMBus(1) for Revision 2 boards
        self.address = 0x68       # This is the address value read via the i2cdetect command
        # Now wake the 6050 up as it starts in sleep mode
        self.bus.write_byte_data(self.address, self.power_mgmt_1, 0)
        self.angular_velocity = 0.0
        self.linear_velocity = 0.0
        self.max_vel = 1.0
        self.iter = 10
        self.delay = 0.0001

    def read_byte(self, adr):
        return self.bus.read_byte_data(self.address, self.adr)

    def read_word(self, adr):
        high = self.bus.read_byte_data(self.address, adr)
        low = self.bus.read_byte_data(self.address, adr+1)
        val = (high << 8) + low
        return val

    def read_word_2c(self, adr):
        val = self.read_word(adr)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def dist(self, a,b):
        return math.sqrt((a*a)+(b*b))

    def get_y_rotation(self, x, y, z):
        radians = math.atan2(x, self.dist(y,z))
        return -math.degrees(radians)

    def get_x_rotation(self, x, y, z):
        radians = math.atan2(y, self.dist(x,z))
        return math.degrees(radians)

    def get_data(self):

        gyro_xout = self.read_word_2c(0x43) / 131
        gyro_yout = self.read_word_2c(0x45) / 131
        gyro_zout = self.read_word_2c(0x47) / 131

        accel_xout = self.read_word_2c(0x3b) / 16384.0
        accel_yout = self.read_word_2c(0x3d) / 16384.0
        accel_zout = self.read_word_2c(0x3f) / 16384.0

        x_rotation = self.get_x_rotation(accel_xout, accel_yout, accel_zout)
        y_rotation = self.get_y_rotation(accel_xout, accel_yout, accel_zout)

        return [gyro_xout, gyro_yout, gyro_zout, accel_xout, accel_yout, accel_zout, x_rotation, y_rotation]

    def get_vels(self):

        last_time = time.time()
        gyro_arr = np.zeros([self.iter])
        for i in range(self.iter):
            gyro_xout, gyro_yout, gyro_zout, accel_xout, accel_yout, accel_zout, x_rotation, y_rotation = self.get_data()
	    dt = time.time() - last_time
            last_time = time.time()
            if np.abs(accel_yout) < 0.05:
	        accel_yout = 0.0
            self.linear_velocity += dt*accel_yout
            gyro_arr[i] = gyro_zout
            time.sleep(self.delay)

        self.angular_velocity = np.mean(gyro_arr)
        self.linear_velocity = min(max(self.linear_velocity, -self.max_vel ), self.max_vel)

        return {'linear_velocity':self.linear_velocity, 'angular_velocity':self.angular_velocity}
