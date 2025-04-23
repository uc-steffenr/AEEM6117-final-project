from typing import List
import matplotlib.pyplot as plt
from animation.animation import BallBeamAnimation
from animation.data_plot import DataPlotter


class Plots:
    """Update both the animation and the plots"""

    def __init__(self) -> None:
        """axs [0] = animation axis
        axs [1], axs[2], axs[3] = z,theta,f"""

        self.fig, self.axs = plt.subplots(3, 2)
        self.fig.delaxes(self.axs[0][1])
        self.fig.delaxes(self.axs[2][1])

        self.anim = BallBeamAnimation()
        self.dataplot = DataPlotter()

        self.anim.ax = self.axs[1, 1]
        self.dataplot.z_ax = self.axs[0, 0]
        self.dataplot.theta_ax = self.axs[1, 0]
        self.dataplot.force_ax = self.axs[2, 0]

    def update(self, time, ref, states, outputs):
        self.anim.update(states)
        self.dataplot.update(time, ref, states, outputs)
        plt.pause(0.001)
