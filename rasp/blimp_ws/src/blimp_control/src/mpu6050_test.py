#!/usr/bin/env python

import mpu6050

imu = mpu6050.MPU6050

if __name__ == '__main__':
    while True:
    	print(imu.get_vels())
