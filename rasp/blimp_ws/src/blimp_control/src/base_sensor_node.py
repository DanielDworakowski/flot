import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Image

class SensorNode():

	def __init__(self, topic_name, msg_type, queue_size=10):
		self.publisher = rospy.Publisher(topic_name, msg_type, queue_size)
		rospy.init_node(topic_name+"_node", anonymous=True)
		self.rate = rospy.Rate(10)
		
	def run():
		pass 
