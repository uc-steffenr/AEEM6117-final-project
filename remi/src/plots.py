from typing import List
import matplotlib.pyplot as plt

# from animation.animation import BallBeamAnimation
from .data_plotter import DataPlotter  # type:ignore


class Plots:
    """Update both the animation and the plots"""

    def __init__(self) -> None:
        self.fig, self.axsd = plt.subplot_mosaic(
            [
                ["thetas", "taus", "anim", "anim"],
                ["theta1", "tau1", "anim", "anim"],
                ["theta2", "tau2", "anim", "anim"],
                ["thetat", "ee", "anim", "anim"],
            ]
        )
        self.fig.tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.state("zoomed")  # show fullscreen
        self.dataplot = DataPlotter(axs=self.axsd)

    def update(self, time, states, outputs):
        # self.anim.update(states)
        self.dataplot.update(time, states, outputs)
        plt.pause(0.001)
