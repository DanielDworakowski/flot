#!/usr/bin/env python
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import mpu6050
import serial
import time
import math

ard_serial = serial.Serial('/dev/ttyACM0',115200)
time.sleep(2)

class TwoPlot():

    def __init__(self):
        self.data_length = 100
        self.ys0 = [0.0]*self.data_length
        self.ys1 = [0.0]*self.data_length
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0,self.data_length])
        self.ax.set_ylim([-5,5])
        self.line0, = self.ax.plot(np.zeros(self.data_length))
        self.line1, = self.ax.plot(np.zeros(self.data_length))
        self.fig.canvas.draw()
        plt.show(block=False)

    def update(self,data):
        self.ys0.pop(0)
        self.ys1.pop(0)
        self.ys0.append(data[0])
        self.ys1.append(data[1])
        self.line0.set_ydata(np.array(self.ys0))
        self.line1.set_ydata(np.array(self.ys1))
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line0)
        self.ax.draw_artist(self.line1)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

imu = mpu6050.MPU6050()

if __name__ == '__main__':
    simple_plotter = TwoPlot()
    while True:
        try:
            time.sleep(0.1)
            w_ard = float(ard_serial.readline().split(',')[0])
            w_imu = imu.get_vels()['angular_velocity']*(math.pi/180.)
            print('Pot: {}').format(w_ard)
            print('IMU: {}').format(w_imu)
            simple_plotter.update([w_ard, w_imu])
        except:
            pass
