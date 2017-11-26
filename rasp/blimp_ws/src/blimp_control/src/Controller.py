# Controller library
import PID

# Base Controller Class
class Controller:
    def __init__(self):

        # Left actuator controller signals
        self.left_ref = 0
        self.left_meas = 0
        self.left_error = 0
        self.left_u = 0

        # Right actuator controller signals
        self.right_ref = 0
        self.right_meas = 0
        self.right_error = 0
        self.right_u = 0

        # Down actuator controller signals
        self.down_ref = 0
        self.down_meas = 0
        self.down_error = 0
        self.down_u = 0


# JP Controller Architecture
# Individual PID controllers for each actuator
# See architecture from: https://drive.google.com/file/d/1ue7TJ0mj9_Mwo_yER_wDqp-s8IjADLwf/view?usp=sharing

class JP_Controller(Controller):
    def __init__(self, diameter=0.1016, left_sat=1., right_sat=1., down_sat=1.):
        super(JP_Controller, self).__init__()

        # Controller output saturation values
        self.left_sat = left_sat
        self.right_sat = right_sat
        self.down_sat = down_sat

        # Individual PID controller
        self.left_PID = PID.PID(min_=-self.left_sat, max_=self.left_sat)
        self.right_PID = PID.PID(min_=-self.right_sat, max_=self.right_sat)
        self.down_PID = PID.PID(min_=-self.down_sat, max_=self.right_sat)

        self.diameter = diameter    # distance between left & right actuators in meters

        print('Initialized JP_PID controller')

    # Config instance variable has: {p, i, d, min, max}
    def configure(self, leftConfig=None, rightConfig=None, downConfig=None):
        # Configure Left/Down/Right PID controllers here
        try:
            if leftConfig is not None:
                self.left_PID.setMinMax(leftConfig.min, leftConfig.max)
                self.left_PID.setKp(leftConfig.p)
                self.left_PID.setKi(leftConfig.i)
                self.left_PID.setKd(leftConfig.d)

            if rightConfig is not None:
                self.right_PID.setMinMax(rightConfig.min, rightConfig.max)
                self.right_PID.setKp(rightConfig.p)
                self.right_PID.setKi(rightConfig.i)
                self.right_PID.setKd(rightConfig.d)

            if downConfig is not None:
                self.down_PID.setMinMax(downConfig.min, downConfig.max)
                self.down_PID.setKp(downConfig.p)
                self.down_PID.setKi(downConfig.i)
                self.down_PID.setKd(downConfig.d)

        except:
            print("Improper config instance variables used")

    # Update function for use ControlLoop class
    # Ref - {x - tangential velocity (m/s), w - angular velocity (rad/s) about z-axis}
    # Sonar - distance (m) in z-direction
    # IMU - {x - tangential velocity (m/s), w - angular velocity (rad/s) about z-axis}
    def update(self, ref, ref_stamp, sonar, imu):
        # Update reference values
        [self.left_ref, self.right_ref] = self.kinematic_transform(ref.x, ref.w, self.diameter)
        self.down_ref = ref.z

        # Update measured values
        [self.left_meas, self.right_meas] = self.kinematic_transform(imu.x, imu.w, self.diameter)
        self.down_meas = sonar

        # Update error values
        self.left_error = self.left_ref - self.left_meas
        self.right_error = self.right_ref - self.right_meas
        self.down_error = self.down_ref - self.down_meas

        # Set references
        self.left_PID.setRef(self.left_ref)
        self.right_PID.setRef(self.right_ref)
        self.down_PID.setRef(self.down_ref)

        # Get controller efforts
        self.left_u = self.left_PID.getCmd(self.left_meas)
        self.right_u = self.right_PID.getCmd(self.right_meas)
        self.down_u = self.down_PID.getCmd(self.down_meas)

        return [self.left_u, self.right_u, self.down_u]

    def kinematic_transform(self, tang_vel, ang_vel, diameter):
        # Calculate for Instantaneous Center of Rotation radius
        r = tang_vel/ang_vel
        vel_left = ang_vel*(r - diameter/2)
        vel_right = ang_vel*(r + diameter/2)

        return [vel_left, vel_right]
