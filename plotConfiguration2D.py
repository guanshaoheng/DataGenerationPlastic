import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plotConfiguration2D(epsList):
    """

    :param epsList: List of the strain [00, 11, 01]
    :return:
    """
    n = len(epsList)
    nodeCoordinate = np.zeros([5, 2, n])
    lx, ly, theta = 1*(1+epsList[:, 0]), 1*(1+epsList[:, 1]), np.pi/2.*(1-epsList[:, 2])
    xtemp = ly*np.cos(theta)*0.5
    ytemp = ly*np.sin(theta)*0.5
    # set the nodes coordinates
    nodeCoordinate[0, 0] = -lx*0.5-xtemp
    nodeCoordinate[0, 1] = -ytemp
    nodeCoordinate[1, 0] = lx*0.5-xtemp
    nodeCoordinate[1, 1] = -ytemp
    nodeCoordinate[2, 0] = lx*0.5+xtemp
    nodeCoordinate[2, 1] = ytemp
    nodeCoordinate[3, 0] = -lx*0.5+xtemp
    nodeCoordinate[3, 1] = ytemp
    nodeCoordinate[4, 0] = -lx*0.5-xtemp
    nodeCoordinate[4, 1] = -ytemp

    indexList = range(0, n, n//200 if n>200 else 1)
    nodeCoordinateShort = nodeCoordinate[..., indexList]
    # plot
    fig = plt.figure()
    ax = plt.axes(xlim=(-1., 1), ylim=(-1., 1))
    line, = ax.plot([], [], lw=3)
    # plt.axis('equal')
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = nodeCoordinateShort[:, 0, i]
        y = nodeCoordinateShort[:, 1, i]
        line.set_data(x, y)
        return line,
    anim = FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(indexList), interval=5, blit=True)
    anim.save('./figSav/deformation.gif')
    return





