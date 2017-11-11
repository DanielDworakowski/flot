import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Image

import Blimp
import PID

class ControlLoop():

    # Default settings: queue size = 10, refresh rate = 10 Hz, default Blimp, default PID controller
    # Default topics: sonar = 'hr_sr04', imu = 'bno055'
    # Blimp and controller objects can be custom defined and passed as constructor argument
    def __init__(self, node_name, queue_size=10, refresh_rate=10, sonar_topic='hr_sr04', imu_topic='bno055',
                 nn_topic='nn_output', blimp=Blimp.Blimp(), controller=PID.PID()):

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

        # Current feedback data
        self.sonar = None
        self.imu = None

        # Attached class objects
        self.blimp = blimp
        self.controller = controller

        # Current reference
        self.ref = None
        self.ref.x = None       # reference for tangential velocity
        self.ref.w = None       # reference for angular velocity
        self.covariance = None  # covariance matrix
        self.ref_stamp = None

        # Current actuation commands
        self.left_cmd = None
        self.right_cmd = None
        self.down_cmd = None

        # Flags
        self.isStarted = False

    def sonar_callback(self, msg):
        self.sonar = msg.data

    def imu_callback(self, msg):
        self.imu = msg.data

    def nn_callback(self, msg):
        self.ref.x = msg.twist.twist.linear.x
        self.ref.w = msg.twist.twist.angular.z
        self.covariance = msg.twist.covariance
        self.ref_stamp = msg.header.stamp

    def start(self):
        self.isStarted = True

        # Main controller loop
        while self.isStarted:

            # If connection is lost, attempt to reconnect in loop iteration
            if not self.blimp.is_connected():
                print('Not connected to blimp, attempting connection...')
                try:
                    self.blimp.connect()
                except:
                    print('Failed to connect to blimp')

            # Get latest actuation values from controller
            [self.left_cmd, self.right_cmd, self.down_cmd] = \
                self.controller.update(self.ref, self.ref_stamp, self.sonar, self.imu)

            # Send new actuation values
            self.blimp.left(self.denomalize(self.left_cmd)
                            if -32768 < self.denomalize(self.left_cmd) < 32768
                            else 0)
            self.blimp.right(self.denomalize(self.right_cmd)
                             if -32768 < self.denomalize(self.right_cmd) < 32768
                             else 0)
            self.blimp.down(self.denomalize(self.down_cmd)
                            if -32768 < self.denomalize(self.down_cmd) < 32768
                            else 0)

    def stop(self):
        self.isStarted = False
        if self.blimp.is_connected():
            self.blimp.stop()

    # Helper functions
    def denomalize(self, value):
        return int(round(value*32767))

