import ai2thor.controller
import numpy as np
import time

class AI2THOR():
    controller = ai2thor.controller.Controller()
    controller.start(player_screen_width=640, player_screen_height=480)
    
    def __init__(self, scene='FloorPlan230', grid_size=0.01, v_rate=0.1, w_rate=0.1, dt = 0.05):
        # Member variables.
        self.controller.reset(scene)
        self.event = self.controller.step(dict(action='Initialize', gridSize=grid_size))
        self.grid_size = grid_size
        self.position = None
        self.rotation = None
        self.yaw = None
        self.image_png = None
        self.image_rgb = None
        self.collided = None
        self.v = 0.
        self.w = 0.
        self.v_rate = v_rate
        self.w_rate = w_rate
        self.dt = dt

    def getPosition(self):
        self.update()
        return self.position

    def getRotation(self):
        self.update()
        return self.rotation

    def getYaw(self):
        self.update()
        return self.yaw

    def getPNGImage(self):
        self.update()
        return self.image_png
        
    def getRGBImage(self):
        self.update()
        return self.image_rgb

    def isCollided(self):
        self.event = self.controller.step(dict(action='MoveAhead'))
        self.update()
        return self.collided

    def update(self):
        self.event = self.controller.step(dict(action='Initialize'))
        self.rotation = self.event.metadata['agent']['rotation']
        self.position = self.event.metadata['agent']['position']
        self.image_png = self.event.image
        self.image_rgb = self.event.frame
        self.yaw = self.event.metadata['agent']['rotation']['y']*(np.pi/180.)
        self.collided = self.event.metadata['collided']

    def takeMultipleSteps(self, steps, axis):
        step_dict = {
            ('x',True): "MoveRight",
            ('x', False): "MoveLeft",
            ('z', True): "MoveAhead",
            ('z', False): "MoveBack"
        }
        is_positive = (steps >= 0)

        success = True
        for i in range(abs(steps)):
            self.event = self.controller.step(dict(action=step_dict[(axis,is_positive)]))
            success = self.event.metadata['lastActionSuccess']
            time.sleep(0.5)
        return success

    # def step(self, v_ref, w_ref):
    #     self.update()
    #     self.v = (1-self.v_rate)*self.v + self.v_rate*v_ref
    #     self.w = (1-self.w_rate)*self.w + self.w_rate*w_ref
    #     new_x = self.position['x'] + self.v*self.dt*np.sin(self.yaw)
    #     new_z = self.position['z'] + self.v*self.dt*np.cos(self.yaw)
    #     new_yaw = self.yaw + self.dt*self.w
    #     self.event = self.controller.step(dict(action='Teleport', x=new_x, y=0.3, z=new_z))
    #     self.event = self.controller.step(dict(action='Rotate', rotation=new_yaw*(180./np.pi)))
    #     print(self.event.metadata['agent']['position'])

    def step(self, v_ref, w_ref):
        # self.update()
        self.v = (1-self.v_rate)*self.v + self.v_rate*v_ref
        self.w = (1-self.w_rate)*self.w + self.w_rate*w_ref
        x_steps = int(self.v*self.dt*np.sin(self.yaw)/self.grid_size)
        z_steps = int(self.v*self.dt*np.cos(self.yaw)/self.grid_size)
        yaw = self.yaw + self.dt*self.w
        x_fail = not self.takeMultipleSteps(x_steps, 'x')
        z_fail = not self.takeMultipleSteps(z_steps, 'z')
        self.collided = x_fail and z_fail
        self.controller.step(dict(action='Rotate', rotation=yaw*(180./np.pi)))
        print(self.collided)
