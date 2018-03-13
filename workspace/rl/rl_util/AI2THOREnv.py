import numpy as np
import matplotlib.pyplot as plt
import environment.AI2THOR as ai2thor

class Env():

    def __init__(self):
        self.env = ai2thor.AI2THOR()
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.getRGBImage()
        self.observation_shape = self.image.shape
        self.action_shape = (2,)

    def step(self, action, render):
        self.image, self.reward, self.done = self.env.step(action)
        return self.image, self.reward, self.done

    def reset(self):
        self.image = self.env.reset()
        return self.image