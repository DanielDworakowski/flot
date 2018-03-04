#!/usr/bin/env python
from rospy import Publisher, init_node, Rate, is_shutdown, ROSInterruptException, get_rostime, Time
from std_msgs.msg import Float64
from blimp_control.msg import Float64WithHeader
import RPi.GPIO as GPIO
import time
import mpu6050
from sensor_msgs.msg import Imu
import math
from tf.transformations import quaternion_from_euler

imu_t = mpu6050.MPU6050()
imu_data = Imu()

def imu():
        #define publisher
        pub0 = Publisher('imu_data', Imu, queue_size=10)
        pub1 = Publisher('yaw_rate', Float64WithHeader, queue_size=10)
        pub2 = Publisher('yaw_rate_control', Float64, queue_size=10)
        init_node('imu', anonymous=True)
        rate = Rate(10) # 10hz
        print('Running IMU node.')
        while not is_shutdown():
            # 
            # Read IMU data.
            data  = imu_t.get_data()
            imu_data.header.stamp = Time.now()
            imu_data.header.stamp = get_rostime()
            # 
            # Get quaternion.
            roll = 180 * math.atan(data[3] / math.sqrt(data[4]**2 + data[5]**2)) / math.pi
            pitch = 180 * math.atan(data[4] / math.sqrt(data[3]**2 + data[5]**2)) / math.pi
            yaw = 180 * math.atan(data[5] / math.sqrt(data[3]**2 + data[5]**2)) / math.pi
            quaternion = quaternion_from_euler(roll,pitch,yaw)
            # 
            # Fill message.
            imu_data.orientation.w = quaternion[0]
            imu_data.orientation.x = quaternion[1]
            imu_data.orientation.y = quaternion[2]
            imu_data.orientation.z = quaternion[3]
            imu_data.linear_acceleration.x = data[3]
            imu_data.linear_acceleration.y = data[4]
            imu_data.linear_acceleration.z = data[5]
            imu_data.linear_acceleration_covariance[0] = -1
            imu_data.angular_velocity.x = data[0]
            imu_data.angular_velocity.y = data[1]
            imu_data.angular_velocity.z = data[2]
            imu_data.angular_velocity_covariance[0] = -1
            pub0.publish(imu_data)

            yaw_rate = Float64WithHeader()
            yaw_rate.header.stamp = get_rostime()
            yaw_rate.float.data = data[2]
            pub1.publish(yaw_rate)
            pub2.publish(data[2]/250.0) #data2 is yaw rate, 1 is pitch rate, 0 roll, have to fix the imudata
            # loginfo(imu_data)
            # loginfo(yaw_rate)
            rate.sleep()

if __name__ == '__main__':
    try:
        imu()
    except ROSInterruptException:
        pass
