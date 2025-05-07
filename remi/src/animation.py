from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Polygon, Wedge
from pydantic import BaseModel, field_validator, ConfigDict
import matplotlib.path as mpath
import numpy as np
from numpy.typing import NDArray


class SMSSat(BaseModel):
    # class SMSSat:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verticies: NDArray[np.float64]
    theta1: float  # FIXME
    theta2: float  # FIXME
    arm_xs: List[float]
    arm_ys: List[float]

    @field_validator("verticies", mode="before")
    def validate_array(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be a numpy ndarray")
        if v.dtype != np.float64:
            raise ValueError("Array must have dtype float64")
        return v

    def draw_base(self, ax: plt.Axes):
        base = Polygon(self.verticies, facecolor="#7CBCFF", ec="white")
        ax.add_patch(base)

    def draw_manipulator(self, ax: plt.Axes):
        arm = Line2D(self.arm_xs, self.arm_ys, color="white", marker=".")
        ax.add_line(arm)

    def draw_sms(self, ax: plt.Axes):
        # ax.set_aspect("equal")
        self.draw_base(ax)
        self.draw_manipulator(ax)
        ax.get_figure().canvas.draw()


class TargetSat(BaseModel):
    # class TargetSat:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verticies: NDArray[np.float64]
    target_x: float
    target_y: float

    @field_validator("verticies", mode="before")
    def validate_array(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be a numpy ndarray")
        if v.dtype != np.float64:
            raise ValueError("Array must have dtype float64")
        return v

    def draw_target(self, ax: plt.Axes):
        # ax.set_aspect("equal")
        target = Polygon(self.verticies, facecolor="#FF5D65", ec="white")
        m = MarkerStyle("*", fillstyle="full")
        ax.plot(
            self.target_x,
            self.target_y,
            marker=m,
            markerfacecolor="tab:green",
            markeredgecolor="white",
        )
        ax.add_patch(target)
        ax.get_figure().canvas.draw()


class SMSTargetAnim:
    def __init__(self, ax: plt.Axes, params: Dict[str, float]):
        self.ax = ax
        self.params = params

    def draw_sms(self, verts, arm_xs, arm_ys, theta1=0, theta2=0):
        sms = SMSSat(
            verticies=verts, theta1=theta1, theta2=theta2, arm_xs=arm_xs, arm_ys=arm_ys
        )
        sms.draw_sms(self.ax)

    def draw_target(self, verts, target_x, target_y):
        target = TargetSat(verticies=verts, target_x=target_x, target_y=target_y)
        target.draw_target(self.ax)

    def update(self, states):
        rho = self.params["rho"]
        r_s = self.params["r_s"]
        r_t = self.params["r_t"]

        C = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        r_0 = lambda th_s: r_s + C(th_s) @ np.array([rho[0], 0.0])
        r_1 = lambda th_s, th_1: r_0(th_s) + C(th_s) @ C(th_1) @ np.array(
            [2.0 * rho[1], 0.0]
        )
        r_2 = lambda th_s, th_1, th_2: r_1(th_s, th_1) + C(th_s) @ C(th_1) @ C(
            th_2
        ) @ np.array([2.0 * rho[2], 0.0])
        r_c = lambda th: r_t + C(th) @ np.array([0.0, rho[3]])

        main_sat_verts = np.array(
            [
                [-rho[0], -rho[0]],
                [rho[0], -rho[0]],
                [rho[0], rho[0]],
                [-rho[0], rho[0]],
                [-rho[0], -rho[0]],
            ]
        )

        targ_sat_verts = np.array(
            [
                [-rho[3], -rho[3]],
                [rho[3], -rho[3]],
                [rho[3], rho[3]],
                [-rho[3], rho[3]],
                [-rho[3], -rho[3]],
            ]
        )

        th_s = states.item(0)
        th_1 = states.item(1)
        th_2 = states.item(2)
        th_t = states.item(3)

        r0 = r_0(th_s)
        r1 = r_1(th_s, th_1)
        r2 = r_2(th_s, th_1, th_2)
        rc = r_c(th_t)

        self.ax.set_aspect("equal")
        main_verts = r_s[:, None] + C(th_s) @ main_sat_verts.T
        targ_verts = r_t[:, None] + C(th_t) @ targ_sat_verts.T
        arm_xs = [r0[0], r1[0], r2[0]]
        arm_ys = [r0[1], r1[1], r2[1]]
        target_pt = [rc[0], rc[1]]

        self.ax.clear()
        self.draw_sms(main_verts.T, arm_xs, arm_ys)
        self.draw_target(targ_verts.T, target_pt[0], target_pt[1])
