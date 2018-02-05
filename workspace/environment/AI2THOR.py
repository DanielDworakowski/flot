import ai2thor.controller

class AI2THOR():
    controller = ai2thor.controller.Controller()
    controller.start()
    
    def __init__(self, scene='FloorPlan28', grid_size=0.05, v_rate=0.1, w_rate=0.1):
        # Member variables.
        self.controller.reset(scene)
        self.event = self.controller.step(dict(action='Initialize', gridSize=grid_size))
        self.position = None
        self.image_png = None
        self.image_rgb = None
        self.collided = None
        self.v = 0.
        self.w = 0.
        self.v_rate = v_rate
        self.w_rate = w_rate

    def getPosition(self):
        self.update()
        return self.position

    def getPNGImage(self):
        self.update()
        return self.image_png
        
    def getPosition(self):
        self.update()
        return self.image_rgb

    def collided(self):
        self.update()
        return self.collided

    def update(self):
        self.event = self.controller.step(dict(action='Initialize'))
        self.position = self.event.metadata['agent']['position']
        self.image_png = self.event.image
        self.image_rgb = self.event.frame
        self.collided = self.event.metadata['collided']

    def step(self, v_ref, w_ref):
        self.update()
        self.v = (1-self.v_rate)*self.v + self.v_rate*self.v_ref
        self.w = (1-self.w_rate)*self.w + self.w_rate*self.w_ref
        self.update()
