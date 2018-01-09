#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import RPi.GPIO as GPIO
import time
import mpu6050
from sensor_msgs.msg import Imu
import math
from tf.transformations import quarternion_from_euler


imu = mpu6050.MPU6050()
imu_data = Imu()

def imu():
        #define publisher
        pub = rospy.Publisher('imu_data', Imu, queue_size=10) #how do you set the q size
        rospy.init_node('imu', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            # [gyro_xout, gyro_yout, gyro_zout, accel_xout, accel_yout, accel_zout, x_rotation, y_rotation]
            data  = imu.get_data()
            # imu_data.header.stamp = rospy.Time.now()
            roll = 180 * math.atan(data[3] / math.sqrt(data[4]**2 + data[5]**2)) / math.pi
            pitch = 180 * math.atan(data[4] / math.sqrt(data[3]**2 + data[5]**2)) / math.pi
            yaw = 180 * math.atan(data[5] / math.sqrt(data[3]**2 + data[5]**2)) / math.pi
            quaternion = quarternion_from_euler(roll,pitch,yaw)
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
            imu_data.orientation.w = quaternion[0]
            rospy.loginfo(imu_data)
            pub.publish(imu_data)
            rate.sleep()

if __name__ == '__main__':
	try:
		imu()
    	except rospy.ROSInterruptException:
        	pass
