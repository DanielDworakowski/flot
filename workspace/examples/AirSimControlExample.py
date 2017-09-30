import AirSimControl
import time

""" Test file to show how to use the AirSimControl module """

with AirSimControl.AirSimControl() as asc:
	asc.setPose(1,1,1,0,0,0)
	for i in range(200):
		asc.setCommand(0.5,3.)
		time.sleep(1/30.)
	for i in range(200):
		asc.setCommand(0.5,-3.)
		time.sleep(1/30.)
	for i in range(200):
		asc.setCommand(2.,0.)
		time.sleep(1/30.)
