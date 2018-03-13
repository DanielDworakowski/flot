import numpy as np
import matplotlib.pyplot as plt

class Env():

    def __init__(self):
        self.env_name = "RoboschoolInvertedPendulum-v1"
        self.env = gym.make(self.env_name)
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.render("rgb_array")
        self.observation_shape = self.env.observation_space.shape
        # self.observation_shape = self.image.shape
        self.action_shape = self.env.action_space.shape
        self.env = gym.wrappers.Monitor(self.env, "/home/rae/videos/"+self.env_name)

    def step(self, action, render):
        action = np.clip(action,-1,1)
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.image = self.env.render("rgb_array")
        self.image = self.state
        return self.image, self.reward, self.done

    def reset(self):
        self.state = self.env.reset()
        self.image = self.env.render("rgb_array")
        self.image = self.state
        return self.image