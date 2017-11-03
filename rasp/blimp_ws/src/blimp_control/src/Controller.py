import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Image

class Controller():

	def __init__(self, node_name, msg_type, queue_size=10):
		rospy.init_node(node_name, anonymous=True)
		self.rate = rospy.Rate(10)
		self.hr_sr04_msg = None 
		self.bno055_msg = None
		
	def hr_sr04_callback(data):

