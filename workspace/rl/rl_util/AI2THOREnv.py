import numpy as np
import matplotlib.animation as animation
from pylab import *
import environment.AI2THOR as ai2thor
plt.rcParams.update({'figure.max_open_warning': 0})

class Env():

    def __init__(self):
        self.env = ai2thor.AI2THOR()
        self.state = self.env.reset()
        self.reward = None
        self.done = False
        self.image = self.env.getRGBImage()
        self.last_image = self.image
        self.state = self.env.getState()
        self.aux_shape = self.state.shape
        self.observation_shape = self.image.shape
        self.action_shape = (2,)
        self.video_imgs = []
        self.vid_dir = "/home/rae/videos/AI2THOR/"


    def step(self, action, render):
        self.last_image = self.image
        self.image, self.reward, self.done, self.state = self.env.step(action)
        if np.sqrt(self.env.episodes).is_integer():
            self.video_imgs.append(self.image)
            if self.done:
                self.save_video()
                self.video_imgs = []
        return (self.image - self.last_image), self.reward, self.done, self.state

    def reset(self):
        self.image = self.env.reset()
        self.last_image = self.image
        return self.image - self.last_image

    def save_video(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        im = ax.imshow(self.video_imgs[0],cmap='gray',interpolation='nearest')
        im.set_clim([0,1])
        fig.set_size_inches([6.8,4.8])

        tight_layout()

        def update_img(n):
            im.set_data(self.video_imgs[n])
            return im

        ani = animation.FuncAnimation(fig,update_img,len(self.video_imgs),interval=20)
        writer = animation.writers['ffmpeg'](fps=20)

        ani.save(self.vid_dir+"episode_"+str(self.env.episodes)+".mp4",writer=writer,dpi=100)
        return ani