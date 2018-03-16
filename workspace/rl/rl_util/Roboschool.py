from OpenGL import GLU
import gym
import roboschool
import numpy as np
import matplotlib.pyplot as plt
import pdb

class Env():

    def __init__(self):
        self.env_name = "RoboschoolInvertedPendulum-v1"
        self.env = gym.make(self.env_name)
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.render("rgb_array")
        self.last_image = self.image
        self.aux_shape = self.env.observation_space.shape
        self.observation_shape = self.image.shape
        self.action_shape = self.env.action_space.shape
        self.env = gym.wrappers.Monitor(self.env, "/home/rae/videos/"+self.env_name, force=True)

    def step(self, action, render):
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.last_image = self.image
        self.image = self.env.render("rgb_array")
        return (self.image - self.last_image), self.reward, self.done, self.state

    def reset(self):
        self.state = self.env.reset()
        self.image = self.env.render("rgb_array")
        self.last_image = self.image
        return self.image - self.last_image