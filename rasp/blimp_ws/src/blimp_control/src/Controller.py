# Placeholder for Controller library
import PID

# JP Controller Architecture
# Individual PID controllers for each actuator
# See architecture from: https://drive.google.com/file/d/1ue7TJ0mj9_Mwo_yER_wDqp-s8IjADLwf/view?usp=sharing

class JP_Controller:
    def __init__(self):
        self.leftPID = PID.PID()
        self.rightPID = PID.PID()
        self.downPID = PID.PID()

        print('Initialized JP_PID controller')

    def configure(self):
        print('Configure placeholder')

    def update(self, ref, ref_stamp, sonar, imu):
        print('Updated')
        return [0.5, 0.5, 0.5]

