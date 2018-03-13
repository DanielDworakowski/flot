from OpenGL import GLU
import gym
import roboschool
import numpy as np
import matplotlib.pyplot as plt

class Env():

    def __init__(self):
        self.env = gym.make("RoboschoolInvertedPendulum-v1")
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.render("rgb_array")
        self.observation_shape = self.env.observation_space.shape
        # self.observation_shape = self.image.shape
        self.action_shape = self.env.action_space.shape

    def step(self, action, render):
        action = np.clip(action,-1,1)
        self.state, self.reward, self.done, _ = self.env.step(action)
        if render:
            self.image = self.env.render("rgb_array")
            plt.imshow(self.image)
            plt.show()
        self.image = self.state
        return self.image, self.reward, self.done

    def reset(self):
        self.state = self.env.reset()
        # self.image = self.env.render("rgb_array").copy()
        self.image = self.state
        return self.image