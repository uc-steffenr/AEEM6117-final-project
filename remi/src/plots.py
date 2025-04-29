from typing import List
import matplotlib.pyplot as plt

# from animation.animation import BallBeamAnimation
from .data_plotter import DataPlotter  # type:ignore


class Plots:
    """Update both the animation and the plots"""

    def __init__(self) -> None:
        self.fig, self.axsd = plt.subplot_mosaic(
            [
                ["thetas", "theta1", "taus", "tau2", "anim", "anim"],
                ["thetat", "theta2", "tau1", "ee", "anim", "anim"],
            ]
        )
        self.fig.tight_layout()

        # self.anim = BallBeamAnimation()
        # self.dataplot = DataPlotter()

        # thetaS_ax = self.axs[0, 0]
        # thetaT_ax = self.axs[1, 0]
        # theta1_ax = self.axs[0, 1]
        # theta2_ax = self.axs[1, 1]
        # tauS_ax = self.axs[0, 2]
        # tau1_ax = self.axs[1, 2]
        # tau2_ax = self.axs[0, 3]
        # ee_ax = self.axs[1, 3]

        # anim_ax = self.axs[:, 4:]

        # self.dataplot = DataPlotter(
        #     [
        #         thetaS_ax,
        #         thetaT_ax,
        #         theta1_ax,
        #         theta2_ax,
        #         tauS_ax,
        #         tau1_ax,
        #         tau2_ax,
        #         ee_ax,
        #         anim_ax,
        #     ]
        # )
        self.dataplot = DataPlotter(axs=self.axsd)

    def update(self, time, states, outputs):
        # def update(self, time, ref, states, outputs):
        # self.anim.update(states)
        self.dataplot.update(time, states, outputs)
        # self.dataplot.update(time, ref, states, outputs)
        plt.pause(0.001)
