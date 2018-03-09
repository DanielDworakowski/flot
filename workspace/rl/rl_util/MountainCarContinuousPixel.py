import gym

class Env():

    def __init__(self):
        self.env = gym.make("MountainCarContinuous-v0")
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.render("rgb_array").copy()
        self.observation_shape = self.image.shape
        self.action_shape = self.env.action_space.shape

    def step(self, action):
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.image = self.env.render("rgb_array").copy()
        return self.image, self.reward, self.done

    def reset(self):
        self.env.reset()
        self.image = self.env.render("rgb_array").copy()
        return self.image