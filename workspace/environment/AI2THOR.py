import ai2thor_src.ai2thor.controller
import numpy as np
import time
import ai2thor_map
import random

class AI2THOR():
    screen_w = 680
    screen_h = 480
    controller = ai2thor_src.ai2thor.controller.BFSController()
    controller.start(player_screen_width=screen_w, player_screen_height=screen_h)
    
    def __init__(self, scene='FloorPlan201', grid_size=0.05, v_rate=0.2, w_rate=0.2, dt = 0.4):
        # Member variables.
        self.observation_shape = (self.screen_h, self.screen_w, 3)
        self.action_shape = (2,)
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
        self.raw_maps = ai2thor_map.occup_grid_dict
        self.grid_scale = 100
        self.floor_plans = []
        self.floor_plan = scene
        self.occup_maps = {}
        self.rawMaptoOccupGrid()
        print(self.occup_maps.keys())
        self.episodes = 0
        self.timestep = 0

    def reset(self):
        self.v = 0.
        self.w = 0.
        self.update()
        # self.floor_plan = random.choice(list(self.occup_maps.keys()))
        grid_x, grid_z = random.choice(list(self.occup_maps[self.floor_plan]))       
        self.event = self.controller.step(dict(action='Teleport', x=grid_x/self.grid_scale*1., y=self.position['y'], z=grid_z/self.grid_scale*1.))
        new_yaw = random.random()*2*3.14
        self.event = self.controller.step(dict(action='Rotate', rotation=new_yaw*(180./np.pi)))
        self.update()
        return self.getRGBImage()


    def rawMaptoOccupGrid(self):
        for floor_plan_name, floor_plan in self.raw_maps.items():
            self.floor_plans.append(floor_plan_name)
            self.occup_maps[floor_plan_name] = set([])
            for free_space in floor_plan:
                grid_x = int(round(free_space['x']*self.grid_scale))
                grid_z = int(round(free_space['z']*self.grid_scale))
                self.occup_maps[floor_plan_name].add((grid_x, grid_z))
        if not(self.floor_plan in self.floor_plans):
            print("Scene not available")

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
        return self.collided

    def update(self):
        self.event = self.controller.step(dict(action='Initialize'))
        self.rotation = self.event.metadata['agent']['rotation']
        self.position = self.event.metadata['agent']['position']
        self.image_png = self.event.image
        self.image_rgb = self.event.frame
        self.yaw = self.event.metadata['agent']['rotation']['y']*(np.pi/180.)

    def positionToGrid(self, position):
        return int(round(int(round(position*self.grid_scale))/(self.grid_size*self.grid_scale))*(self.grid_size*self.grid_scale))

    def step(self, action):
        action = np.clip(action, -1, 1)
        v_ref, w_ref = action[0], action[1]
        self.update()
        self.v = (1-self.v_rate)*self.v + self.v_rate*v_ref
        self.w = (1-self.w_rate)*self.w + self.w_rate*w_ref
        new_x = self.position['x'] + self.v*self.dt*np.sin(self.yaw)
        new_z = self.position['z'] + self.v*self.dt*np.cos(self.yaw)
        new_yaw = self.yaw + self.dt*self.w
        grid_x = self.positionToGrid(new_x)
        grid_z = self.positionToGrid(new_z)
        success = False
        self.event = self.controller.step(dict(action='Rotate', rotation=new_yaw*(180./np.pi)))
        if (grid_x, grid_z) in self.occup_maps[self.floor_plan]:            
            self.event = self.controller.step(dict(action='Teleport', x=grid_x/self.grid_scale*1., y=self.position['y'], z=grid_z/self.grid_scale*1.))
            success = True
        self.update()
        self.collided = not success
        self.timestep += 1
        if self.timestep > 500:
            self.collided = True
        if self.collided:
            self.episodes += 1
            self.timestep = 0
        return_list = [self.getRGBImage(), self.v, self.collided]
        return return_list
