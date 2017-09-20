import AirSimControl as ctrl
import Observations as observations
import AirSimObservations as obsImpl
import math as m
import time
from PIL import Image
#
# clTest cases for use of control.
observer = obsImpl.AirSimObservations()
obs = observations.Observation()
with ctrl.AirSimControl() as ct:
    # ct.followPathSync(5,5,50)
    # ct.followPathSync(5,5,-50)
    # ct.setPath(5, 50)
    observer.observe(obs)
    print(obs.cameraImageU8.shape)
    # ct.setPath(5, -50)
    time.sleep(3)

print("done")

# while 1:
# # for x in range(30 * 3):
#     w = v_t / r
#     print(w)
#     w_deg = w * 180. / m.pi
#     # yaw = curYaw#ct.client.toEulerianAngle(ct.client.getOrientation())[2]
#     yaw = ct.client.toEulerianAngle(ct.client.getOrientation())[2]
#     v_x = v_t * m.cos(yaw)
#     v_y = v_t * m.sin(yaw)
#     ct.client.moveByVelocity(v_x, v_y, 0, 1, ctrl.DrivetrainType.ForwardOnly, ctrl.YawMode(is_rate=True, yaw_or_rate=w_deg))
#     # ct.client.moveByVelocity(v_x, v_y, 0, 1, ctrl.DrivetrainType.MaxDegreeOfFreedom, ctrl.YawMode(is_rate=False, yaw_or_rate=yaw))
#     print('v_x: %G v_y: %G yaw: %G'%(v_x, v_y, yaw))
#     time.sleep(1/30.)
#     curYaw += 1/30. * w
