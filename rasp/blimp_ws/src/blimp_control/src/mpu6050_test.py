#!/usr/bin/env python

import mpu6050
#import simple_plot

imu = mpu6050.MPU6050()

if __name__ == '__main__':
    #simple_plotter = simple_plot.SimplePlot(2)
    while True:
    	#simple_plotter.update(imu.get_vels().values())
	print(imu.get_vels().values())
