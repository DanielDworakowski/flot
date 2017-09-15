from PythonClient import *
import timeit
# connect to the AirSim simulator
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# client.goHome()
client.takeoff()

client.moveToPosition(-10, 10, -10, 5)
# print(client.moveByVelocity(-10, 0, 0, 5, yaw_mode=YawMode(yaw_or_rate)))
# client.moveByAngle(0,0,5,1,10)
print('getting debug info')
print(client.getRollPitchYaw())
# client.hover()


AirSimClient.wait_key('Press any key to take images')
responses = client.simGetImages([
    ImageRequest(0, AirSimImageType.Scene)])

print('Retrieved images: %d', len(responses))s

for response in responses:
    if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        AirSimClient.write_pfm(os.path.normpath('/home/ddworakowski/flot/sim/testing/py1.pfm'), AirSimClient.getPfmArray(response))
    else:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        AirSimClient.write_file(os.path.normpath('/home/ddworakowski/flot/sim/testing/py1.png'), response.image_data_uint8)
