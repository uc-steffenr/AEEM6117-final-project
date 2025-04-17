"""Defines functions used in CLIK control."""
import numpy as np

from .utils import load_func
from .kinematics import (calc_capture_point_position,
                         calc_capture_point_velocity,
                         calc_critical_positions,
                         calc_obstacle_point_positions,
                         calc_obstacle_point_velocities)


# Load CLIK functions
# ----------------------------------------------------------------------
gam1 = load_func('gam1.pkl')
gam2 = load_func('gam2.pkl')
gam = load_func('gam.pkl')

lam = load_func('lam.pkl')
lam_dot = load_func('lam_dot.pkl')


# Gamma Wrappers
# ----------------------------------------------------------------------
def calc_gamma1(y : np.ndarray,
                p : np.ndarray,
                q_bar : np.ndarray,
                q_max : np.ndarray,
                q_min : np.ndarray) -> float:
    """Calculates Gamma 1 secondary objective.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    p : np.ndarray
        Angle limit weights (positive).
    q_bar : np.ndarray
        Center values of the full range of coordinate q.
    q_max : np.ndarray
        Maximum q value.
    q_min : np.ndarray
        Minimum q value.

    Returns
    -------
    float
        Gamma 1 value.
    """
    return gam1(y, p, q_bar, q_max, q_min)


def calc_gamma2(y : np.ndarray,
                rho : np.ndarray,
                r_s : np.ndarray,
                w : np.ndarray,
                op_x : np.ndarray,
                op_y : np.ndarray) -> float:
    """Calculates Gamma 2 secondary objective.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.
    w : np.ndarray
        Distance weight corresponding to critical point-obstacle point
        pairs. There are 6 of them, and they must be in a 1D array.
        (w11, w12, w21, w22, etc...)
    op_x : np.ndarray
        Obstacle point x-coordinates.
    op_y : np.ndarray
        Obstacle point y-coordinates.

    Returns
    -------
    float
        Gamma 2 value.
    """
    pass


# NOTE: this may cause an error depending on the output of gam
# if so, just mess around with indices until it works
def calc_gamma(y : np.ndarray,
               rho : np.ndarray,
               r_s : np.ndarray,
               p : np.ndarray,
               q_bar : np.ndarray,
               q_max : np.ndarray,
               q_min : np.ndarray,
               w : np.ndarray,
               op_x : np.ndarray,
               op_y : np.ndarray) -> float:
    """Calculates Gamma value.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.
    p : np.ndarray
        Angle limit weights (positive).
    q_bar : np.ndarray
        Center values of the full range of coordinate q.
    q_max : np.ndarray
        Maximum q value.
    q_min : np.ndarray
        Minimum q value.
    w : np.ndarray
        Distance weight corresponding to critical point-obstacle point
        pairs. There are 6 of them, and they must be in a 1D array.
        (w11, w12, w21, w22, etc...)
    op_x : np.ndarray
        Obstacle point x-coordinates.
    op_y : np.ndarray
        Obstacle point y-coordinates.

    Returns
    -------
    float
        Gamma value.
    """
    return gam(y, rho, r_s, p, q_bar, q_max, q_min, w, op_x, op_y)[0, 0]


# Lambda Wrappers
# ----------------------------------------------------------------------
def calc_lambda(y : np.ndarray,
                rho : np.ndarray,
                r_s : np.ndarray,
                p : np.ndarray,
                q_bar : np.ndarray,
                q_max : np.ndarray,
                q_min : np.ndarray,
                w : np.ndarray,
                op_x : np.ndarray,
                op_y : np.ndarray) -> np.ndarray:
    """Calculates lambda.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.
    p : np.ndarray
        Angle limit weights (positive).
    q_bar : np.ndarray
        Center values of the full range of coordinate q.
    q_max : np.ndarray
        Maximum q value.
    q_min : np.ndarray
        Minimum q value.
    w : np.ndarray
        Distance weight corresponding to critical point-obstacle point
        pairs. There are 6 of them, and they must be in a 1D array.
        (w11, w12, w21, w22, etc...)
    op_x : np.ndarray
        Obstacle point x-coordinates.
    op_y : np.ndarray
        Obstacle point y-coordinates.

    Returns
    -------
    np.ndarray
        Lambda value.
    """
    return lam(y, rho, r_s, p, q_bar, q_max, q_min, w, op_x, op_y)


def calc_lambda_dot(y : np.ndarray,
                    rho : np.ndarray,
                    r_s : np.ndarray,
                    p : np.ndarray,
                    q_bar : np.ndarray,
                    q_max : np.ndarray,
                    q_min : np.ndarray,
                    w : np.ndarray,
                    op_x : np.ndarray,
                    op_y : np.ndarray,
                    vx : np.ndarray,
                    vy : np.ndarray) -> np.ndarray:
    """Calculates time derivative of lambda.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.
    p : np.ndarray
        Angle limit weights (positive).
    q_bar : np.ndarray
        Center values of the full range of coordinate q.
    q_max : np.ndarray
        Maximum q value.
    q_min : np.ndarray
        Minimum q value.
    w : np.ndarray
        Distance weight corresponding to critical point-obstacle point
        pairs. There are 6 of them, and they must be in a 1D array.
        (w11, w12, w21, w22, etc...)
    op_x : np.ndarray
        Obstacle point x-coordinates.
    op_y : np.ndarray
        Obstacle point y-coordinates.
    vx : np.ndarray
        Obstacle point x-velocities (1D array).
    vy : np.ndarray
        Obstacle point y-velocities (1D array).

    Returns
    -------
    np.ndarray
        Time derivative of lambda value.
    """
    return lam_dot(y, rho, r_s, p, q_bar, q_max, q_min, w, op_x, op_y, vx, vy)


# Wrapped Functions That Take Care of Obstacle Point Search
# ----------------------------------------------------------------------
def _get_obstacle_points(y : np.ndarray,
                         rho : np.ndarray,
                         r_s : np.ndarray,
                         r_t : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gets the two closest obstacle points for every critical point.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of servicing satellite.
    r_t : np.ndarray
        Central position of target satellite.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Position and velocity arrays.
    """
    # Calculate positions of obstacles and critical points
    ops = list(calc_obstacle_point_positions(y, rho, r_t))
    capture_pt = calc_capture_point_position(y, rho, r_t)
    ops += [capture_pt]
    ops = [op.flatten() for op in ops]

    cps = list(calc_critical_positions(y, rho, r_s))
    cps = [cp.flatten() for cp in cps]

    # Calculate velocities of obstacle points
    v_ops = list(calc_obstacle_point_velocities(y, rho, r_t))
    v_capture_pt = calc_capture_point_velocity(y, rho, r_t)
    v_ops +=[v_capture_pt]
    v_ops = [v.flatten() for v in v_ops]

    # Search for closest obstacle points for each critical point
    pos = np.zeros((3, 2, 2))
    vel = np.zeros((3, 2, 2))
    for i, cp in enumerate(cps):
        dists = [np.sqrt((cp[0] - op[0])**2 + (cp[1] - op[1])**2) for op in ops]
        sorted_dists = sorted(dists, key=float)

        ind1 = dists.index(sorted_dists[0])
        ind2 = dists.index(sorted_dists[1])

        # Make sure the capture point isn't considered for the end effector
        if i == 2 and ind1 == len(ops)-1:
            ind1 = dists.index(sorted_dists[2])
        elif i == 2 and ind2 == len(ops)-1:
            ind2 = dists.index(sorted_dists[2])

        pos[i, 0, :] = ops[ind1]
        pos[i, 1, :] = ops[ind2]
        vel[i, 0, :] = v_ops[ind1]
        vel[i, 1, :] = v_ops[ind2]

    return pos, vel


def calc_clik_params(y : np.ndarray,
                     rho : np.ndarray,
                     r_s : np.ndarray,
                     r_t : np.ndarray,
                     p : np.ndarray,
                     q_bar : np.ndarray,
                     q_max : np.ndarray,
                     q_min : np.ndarray,
                     w : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retrieves proper obstacle points and calculates lambda and lambda_dot
    using those points.

    Parameters
    ----------
    y : np.ndarray
        States of the system.
    rho : np.ndarray
        Dimension array.
    r_s : np.ndarray
        Central position of the servicing satellite.
    r_t : np.ndarray
        Central position of the target satellite.
    p : np.ndarray
        Positive weights for gamma 1.
    q_bar : np.ndarray
        Central angle for allowable angle limits.
    q_max : np.ndarray
        Maximum allowable angle.
    q_min : np.ndarray
        Minimum allowable angle.
    w : np.ndarray
        Weights for gamma 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lambda and lambda_dot values.
    """
    pos, vel = _get_obstacle_points(y, rho, r_s, r_t)

    lam = calc_lambda(y,
                      rho,
                      r_s,
                      p,
                      q_bar,
                      q_max,
                      q_min,
                      w,
                      pos[:, 0, :],
                      pos[:, 1, :])
    lam_dot = calc_lambda_dot(y,
                              rho,
                              r_s,
                              p,
                              q_bar,
                              q_max,
                              q_min,
                              w,
                              pos[:, 0, :],
                              pos[:, 1, :],
                              vel[:, :, 0].flatten(),
                              vel[:, :, 1].flatten())

    return lam, lam_dot
