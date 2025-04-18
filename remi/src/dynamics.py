"""Defines dynamics functions."""
import os
import numpy as np
import cloudpickle
from sympy.utilities.lambdify import lambdify

REMI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')

# Load dynamics functions
# ----------------------------------------------------------------------
with open(os.path.join(REMI_DIR, 'functions', 'forward_dynamics.pkl'), 'rb') as f:
    forward_args, forward_expr = cloudpickle.load(f)

with open(os.path.join(REMI_DIR, 'functions', 'inverse_dynamics.pkl'), 'rb') as f:
    inverse_args, inverse_expr = cloudpickle.load(f)

forward = lambdify(forward_args, forward_expr, 'numpy')
inverse = lambdify(inverse_args, inverse_expr, 'numpy')


# Wrappers
# ----------------------------------------------------------------------
def forward_dynamics(y : np.ndarray,
                     tau : np.ndarray,
                     r_s : np.ndarray,
                     r_t : np.ndarray,
                     rho : np.ndarray,
                     m : np.ndarray,
                     I : np.ndarray,
                     d : np.ndarray) -> np.ndarray:
    """Calculates state time derivative.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    tau : np.ndarray
        Joint torques.
    r_s : np.ndarray
        Central position of the servicing satellite.
    r_t : np.ndarray
        Central position of the target satellite.
    rho : np.ndarray
        Dimension array.
    m : np.ndarray
        Mass array.
    I : np.ndarray
        Inertia array.
    d : np.ndarray
        Disturbance array.

    Returns
    -------
    np.ndarray
        State time derivative.
    """
    return forward(y, tau, r_s, r_t, rho, m, I, d).squeeze()


def inverse_dynamics(y : np.ndarray,
                     ddths : np.ndarray,
                     r_s : np.ndarray,
                     r_t : np.ndarray,
                     rho : np.ndarray,
                     m : np.ndarray,
                     I : np.ndarray,
                     d : np.ndarray) -> np.ndarray:
    """Calculates joint torques given desired acceleration.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    ddths : np.ndarray
        Desired joint acceleration.
    r_s : np.ndarray
        Central position of the servicing satellite.
    r_t : np.ndarray
        Central position of the target satellite.
    rho : np.ndarray
        Dimension array.
    m : np.ndarray
        Mass array.
    I : np.ndarray
        Inertia array.
    d : np.ndarray
        Disturbance array.

    Returns
    -------
    np.ndarray
        Joint torques.
    """
    return inverse(y, ddths, r_s, r_t, rho, m, I, d).squeeze()
