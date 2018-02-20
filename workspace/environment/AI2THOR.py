import ai2thor_src.ai2thor.controller
import numpy as np
import time
import ai2thor_map

class AI2THOR():
    controller = ai2thor_src.ai2thor.controller.BFSController()
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
        self.raw_maps = ai2thor_map.occup_grid_dict
        self.grid_scale = 100
        self.floor_plans = []
        self.floor_plan = scene
        self.occup_maps = {}
        self.rawMaptoOccupGrid()

    def rawMaptoOccupGrid(self):
        for floor_plan_name, floor_plan in self.raw_maps.items():
            self.floor_plans.append(floor_plan_name)
            self.occup_maps[floor_plan_name] = set([])
            for free_space in floor_plan:
                grid_x = int(round(floor_plan['x']*self.grid_scale))
                grid_z = int(round(floor_plan['z']*self.grid_scale))
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

    def step(self, v_ref, w_ref):
        self.update()
        self.v = (1-self.v_rate)*self.v + self.v_rate*v_ref
        self.w = (1-self.w_rate)*self.w + self.w_rate*w_ref
        new_x = self.position['x'] + self.v*self.dt*np.sin(self.yaw)
        new_z = self.position['z'] + self.v*self.dt*np.cos(self.yaw)
        new_yaw = self.yaw + self.dt*self.w
        grid_x = int(round(new_x*self.grid_scale))
        grid_z = int(round(new_z*self.grid_scale))
        success = False
        if (grid_x, grid_z) in self.occup_maps[self.floor_plan]:            
            self.event = self.controller.step(dict(action='Teleport', x=new_x, y=0.3, z=new_z))
            self.event = self.controller.step(dict(action='Rotate', rotation=new_yaw*(180./np.pi)))
            success = True
        return success
