import rospy

from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

def main():

    rospy.init_node('topic_test', anonymous=True)

    rate = rospy.Rate(10)

    # Sensor topics
    sonar_pub = rospy.Publisher('sonar_test', Float32, queue_size=10)
    imu_pub = rospy.Publisher('imu_test', Imu, queue_size=1)
    nn_pub = rospy.Publisher('nn_test', TwistWithCovarianceStamped, queue_size=1)

    # ControlLoop Topic
    cl_pub = rospy.Publisher('control_test', Float32MultiArray, queue_size=1)

    # Sensor Messages
    sonar_msg = Float32()
    sonar_msg.data = 1234

    imu_msg = Imu()
    imu_msg.linear_acceleration.x = 10
    imu_msg.angular_velocity.z = 10

    nn_msg = TwistWithCovarianceStamped()

    cl_msg = Float32MultiArray()
    cl_msg.data = [1.1, 2.2]

    while not rospy.is_shutdown():
        sonar_pub.publish(sonar_msg)
        imu_pub.publish(imu_msg)
        nn_pub.publish(nn_msg)
        cl_pub.publish(cl_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass