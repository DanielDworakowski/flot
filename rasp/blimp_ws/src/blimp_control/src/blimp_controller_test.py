#!/usr/bin/env python

import ControlLoop
import Controller
import Blimp

# Initialize and configure blimp
blimp = Blimp.Blimp()
blimp.connect()

# Initialize and configure controller
# Controller must have update function with the appropriate input/output
controller = Controller.JP_Controller()

control_loop = ControlLoop.ControlLoop('test_loop_output_node',
                                       blimp=blimp,
                                       controller=controller,
                                       queue_size=1,
                                       refresh_rate=10,
                                       sonar_topic='sonar_test',
                                       imu_topic='imu_test',
                                       nn_topic='nn_test')

print('Starting controller loop')
control_loop.start()

print('Stopping controller loop')
control_loop.stop()
