"""Utilities for simulation."""

import os
import numpy as np
from typing import Callable
import dill

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REMI_DIR = os.path.dirname(os.path.dirname(__file__))
print(f"REPO_DIR = {REPO_DIR}\t REMI_DIR = {REMI_DIR}")
print(f"__file__ = {__file__}")


def load_position_func():  # -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],(np.ndarray, np.ndarray, np.ndarray)]:
    """Returns position function for system.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             (np.ndarray, np.ndarray, np.ndarray)]
        Given states y, r_s, r_t, and rho, returns r_1, r_2, and r_c.
    """
    with open(os.path.join(REMI_DIR, "src", "functions", "position.pkl"), "rb") as f:
        return dill.load(f)


def load_velocity_func():  # -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],(np.ndarray, np.ndarray, np.ndarray)]:
    """Returns velocity function for the system

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             (np.ndarray, np.ndarray, np.ndarray)]
        Given states y, r_s, r_t, and rho, returns v_1, v_2, and v_c.
    """
    with open(os.path.join(REMI_DIR, "src", "functions", "velocity.pkl"), "rb") as f:
        return dill.load(f)


def load_dynamics_func():
    with open(os.path.join(REMI_DIR, "src", "functions", "dynamics.pkl"), "rb") as f:
        return dill.load(f)


def load_ee_func() -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Returns function to calculate end effector position.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Given states y, r_s, and rho, returns end effector position in inertial frame.
    """
    with open(os.path.join(_pth, "functions", "ee.pkl"), "rb") as f:
        return dill.load(f)


def load_Je_func() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Returns Jacobian of end effector.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        Given states y and rho, returns end effector Jacobian Je.
    """
    with open(os.path.join(_pth, "functions", "Je.pkl"), "rb") as f:
        return dill.load(f)
