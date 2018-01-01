import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import numpy as np

class SimplePlot():

    def __init__(self, number_of_figures=1):
        self.number_of_figures = number_of_figures
        self.data_length = 100
        self.ys = []
        self.figs = []
        self.axs = []
        self.lines = []
        self.t=range(self.data_length)
        for i in range(self.number_of_figures):
            self.ys.append([0.0]*self.data_length)
            fig, ax = plt.subplots()
            ax.set_xlim([0,self.data_length])
            ax.set_ylim([-100,100])
            line, = ax.plot(np.zeros(self.data_length))
            self.figs.append(fig)
            self.axs.append(ax)
            self.lines.append(line)
            self.figs[i].canvas.draw()
        plt.show(block=False)

    def update(self,data):
        for i in range(self.number_of_figures):
            self.ys[i].pop(0)
            self.ys[i].append(data[i])
            self.lines[i].set_ydata(np.array(self.ys[i]))
            self.axs[i].draw_artist(self.axs[i].patch)
            self.axs[i].draw_artist(self.lines[i])
            self.figs[i].canvas.update()
            self.figs[i].canvas.flush_events()
