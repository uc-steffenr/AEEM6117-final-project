from typing import List
import matplotlib.pyplot as plt
import numpy as np
from .data_plotter import DataPlotter  # type:ignore
from .animation import SMSTargetAnim


class Plots:
    """Update both the animation and the plots"""

    def __init__(self) -> None:
        r_s = np.array([-1.5, 0.0])
        r_t = np.array([1.5, 0.0])
        rho = np.array([0.5, 0.5, 0.5, 0.5])
        m = np.array([250.0, 25.0, 25.0, 180.0])
        I = np.array([25.0, 2.5, 2.5, 18.0])
        d = np.zeros(4)
        parameters = dict(r_s=r_s, r_t=r_t, rho=rho, m=m, I=I, d=d)

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
        # figManager.window.state("zoomed")  # show fullscreen
        self.dataplot = DataPlotter(axs=self.axsd, params=parameters)
        self.anim = SMSTargetAnim(ax=self.axsd["anim"], params=parameters)

    def update(self, time, states, outputs):
        self.anim.update(states)
        self.dataplot.update(time, states, outputs)
        plt.pause(0.001)
