import ai2thor.controller

class AI2THOR():
    controller = ai2thor.controller.Controller()
    controller.start()
    
    def __init__(self, scene='FloorPlan28', grid_size=0.05):
        # Member variables.
        self.controller.reset(scene)
        self.controller.step(dict(action='Initialize', gridSize=grid_size))
