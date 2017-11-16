import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plotTrajectory(obs):
    mpl.rcParams['legend.fontsize']= 14
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    z = [0]
    x = [0]
    y = [0]
    plt.ion()
    #
    #plot data here
    for i in range(obs):
        ax.scatter(obs[i], obs[i+1], obs[i+2])
    #
    #leave window open
    while True:
        plt.pause(0.05)
