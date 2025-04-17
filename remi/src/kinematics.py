"""Defines kinematic functions."""
import numpy as np
from .utils import load_func


# Load kinematic functions
# ----------------------------------------------------------------------
cp = load_func('cp.pkl')        # critical positions
r_c = load_func('r_c.pkl')      # position of capture point
r_ee = load_func('r_ee.pkl')    # end effector position
ops = load_func('ops.pkl')      # obstacle point positions

v_cp = load_func('v_cp.pkl')    # critical position velocities
v_c = load_func('v_c.pkl')      # capture point velocity
v_ee = load_func('v_ee.pkl')    # end effector velocity
v_ops = load_func('v_ops.pkl')  # obstacle point velocities

a_c = load_func('a_c.pkl')      # capture point acceleration
J = load_func('J.pkl')          # end effector Jacobian
J_dot = load_func('J_dot.pkl')  # Jacobian time derivative


# Position Wrappers
# ----------------------------------------------------------------------
def calc_critical_positions(y : np.ndarray,
                            rho : np.ndarray,
                            r_s : np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates critical position points.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of servicing satellite.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Cartesian coordinates of each critical position [joint 2,
        middle of link 2, end effector].
    """
    return cp(y, rho, r_s)


def calc_capture_point_position(y : np.ndarray,
                                rho : np.ndarray,
                                r_t : np.ndarray) -> np.ndarray:
    """Calculates caputre point position.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_t : np.ndarray
        Central position of target satellite.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of capture point position.
    """
    return r_c(y, rho, r_t)


def calc_end_effector_position(y : np.ndarray,
                               rho : np.ndarray,
                               r_s : np.ndarray) -> np.ndarray:
    """Calculates end effector position.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the end effector position.
    """
    return r_ee(y, rho, r_s)


def calc_obstacle_point_positions(y : np.ndarray,
                                  rho : np.ndarray,
                                  r_t : np.ndarray) -> tuple[np.ndarray]:
    """Calculates obstacle point positions.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_t : np.ndarray
        Central position of the target satellite.

    Returns
    -------
    tuple[np.ndarray]
        Cartestian coordinates of the obstacle points (7 of them).
    """
    return ops(y, rho, r_t)


# Velocity Wrappers
# ----------------------------------------------------------------------
def calc_critical_velocities(y : np.ndarray,
                             rho : np.ndarray,
                             r_s : np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates critical position velocities.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of servicing satellite.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Velocity of each critical position [joint 2, middle of link 2,
        end effector].
    """
    return v_cp(y, rho, r_s)


def calc_capture_point_velocity(y : np.ndarray,
                                rho : np.ndarray,
                                r_t : np.ndarray) -> np.ndarray:
    """Calculates capture point's velocity.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_t : np.ndarray
        Central position of target satellite.

    Returns
    -------
    np.ndarray
        Velocty of the capture point.
    """
    return v_cp(y, rho, r_t)


def calc_end_effector_velocity(y : np.ndarray,
                               rho : np.ndarray,
                               r_s : np.ndarray) -> np.ndarray:
    """Calculates end effector velocity.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.

    Returns
    -------
    np.ndarray
        Velocity of the end effector.
    """
    return v_ee(y, rho, r_s)


def calc_obstacle_point_velocities(y : np.ndarray,
                                   rho : np.ndarray,
                                   r_t : np.ndarray) -> tuple[np.ndarray]:
    """Calculates obstacle point velocities.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_t : np.ndarray
        Central position of the target satellite.

    Returns
    -------
    tuple[np.ndarray]
        Velocities of the obstacle points (7 of them).
    """
    return v_ops(y, rho, r_t)


# Acceleration Wrapper
# ----------------------------------------------------------------------
def calc_capture_point_acceleration(y : np.ndarray,
                                    ddth_t : float,
                                    rho : np.ndarray,
                                    r_t : np.ndarray) -> np.ndarray:
    """Calculates caputre point acceleration.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    ddth_t : float
        Angular acceleration of the target.
    rho : np.ndarray
        Dimension array.
    r_t : np.ndarray
        Central position of target satellite.

    Returns
    -------
    np.ndarray
        Acceleration of capture point.
    """
    return a_c(y, ddth_t, rho, r_t)


# Jacobian Wrappers
# ----------------------------------------------------------------------
def calc_J(y : np.ndarray,
           rho : np.ndarray,
           r_s : np.ndarray) -> np.ndarray:
    """Calculates Jacobian of end effector.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.

    Returns
    -------
    np.ndarray
        Jacobian of the end effector.
    """
    return J(y, rho, r_s)


def calc_J_dot(y : np.ndarray,
               rho : np.ndarray,
               r_s : np.ndarray) -> np.ndarray:
    """Calculates time derivative of Jacobian of end effector.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.

    Returns
    -------
    np.ndarray
        Jacobian time derivative of the end effector.
    """
    return J_dot(y, rho, r_s)
