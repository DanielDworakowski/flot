import rospy

from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

from sensor_msgs.msg import Image

import Blimp
import Controller

class ControlLoop():

    # Default settings: queue size = 10, refresh rate = 10 Hz, default Blimp, default PID controller
    # Default topics: sonar = 'hr_sr04', imu = 'bno055'
    # Blimp and controller objects is custom defined and must be passed as constructor argument
    def __init__(self, node_name, blimp, controller, queue_size=1, refresh_rate=10, sonar_topic='hr_sr04', imu_topic='bno055',
                 nn_topic='nn_output', output_topic='blimp_test'):

        # Initiate controller node
        rospy.init_node(node_name, anonymous=True)

        # Set refresh rate in Hz
        self.rate = rospy.Rate(refresh_rate)

        # Subscribe to Sonar topic
        rospy.Subscriber(sonar_topic, Float32, self.sonar_callback, queue_size=queue_size)

        # Subscribe to IMU topic
        rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=queue_size)

        # Subscribe to Neural Network output
        rospy.Subscriber(nn_topic, TwistWithCovarianceStamped, self.nn_callback, queue_size=queue_size)

        # Subscribe to Neural Network output
        self.output_pub = rospy.Publisher(output_topic, Float32MultiArray, queue_size=1)

        # Current feedback data
        self.sonar = None
        self.imu = lambda: 0

        # Attached class objects
        self.blimp = blimp
        self.controller = controller

        # Current reference
        self.ref = lambda: 0    # reference for tangential/angular velocity
        self.ref.x = None       # reference for tangential velocity
        self.ref.w = None       # reference for angular velocity
        self.covariance = None  # covariance matrix
        self.ref_stamp = None

        # Current actuation commands
        self.left_cmd = None
        self.right_cmd = None
        self.down_cmd = None

        # ControlLoop Published Message
        self.output_msg = Float32MultiArray()

        # Flags
        self.isStarted = False

        print('Controller loop initialized')

    def sonar_callback(self, msg):
        # print('New sonar message')
        self.sonar = msg.data

    def imu_callback(self, msg):
        # print('New imu message')
        self.imu = msg

    def nn_callback(self, msg):
        # print('New nn callback')
        self.ref.x = msg.twist.twist.linear.x
        self.ref.w = msg.twist.twist.angular.z
        self.covariance = msg.twist.covariance
        self.ref_stamp = msg.header.stamp

    def start(self):
        self.isStarted = True
        print('Starting Control Loop...')

        # Main controller loop
        while not rospy.is_shutdown():

            # If connection is lost, attempt to reconnect in loop iteration
            if not self.blimp.is_connected():
                print('Not connected to blimp, attempting connection...')
                self.blimp.connect()

            # Get latest actuation values from controller
            # print('Updating actuator commands from controller...')
            [self.left_cmd, self.right_cmd, self.down_cmd] = \
                self.controller.update(self.ref, self.ref_stamp, self.sonar, self.imu)

            # Send new actuation values
            # print('Commanding the blimp...')
            self.blimp.left(self.denomalize(self.left_cmd)
                            if -32768 < self.denomalize(self.left_cmd) < 32768
                            else 0)
            self.blimp.right(self.denomalize(self.right_cmd)
                             if -32768 < self.denomalize(self.right_cmd) < 32768
                             else 0)
            self.blimp.down(self.denomalize(self.down_cmd)
                            if -32768 < self.denomalize(self.down_cmd) < 32768
                            else 0)

            # Assign and publish actuation values
            # Left is at index 0, right is at index 1, down is at index 2
            self.output_msg.data = [self.left_cmd, self.right_cmd, self.down_cmd]
            self.output_pub.publish(self.output_msg)

            # Sleeping to keep with refresh rate
            self.rate.sleep()

    def stop(self):
        print('Stopping Control Loop...')
        self.isStarted = False
        if self.blimp.is_connected():
            self.blimp.stop()

    # Helper functions

    # Normalized values are from -1 <= x <= 1
    # Denormalize to actuation value
    def denomalize(self, value):
        return int(round(value*32767))
