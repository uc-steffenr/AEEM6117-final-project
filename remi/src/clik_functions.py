"""Defines functions used in CLIK control."""
import numpy as np
from .utils import load_func


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
    """_summary_

    Parameters
    ----------
    y : np.ndarray
        _description_
    p : np.ndarray
        _description_
    q_bar : np.ndarray
        _description_
    q_max : np.ndarray
        _description_
    q_min : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """
    return gam1(y, p, q_bar, q_max, q_min)


def calc_gamma2(y : np.ndarray,
                rho : np.ndarray,
                r_s : np.ndarray,
                w : np.ndarray,
                op_x : np.ndarray,
                op_y : np.ndarray) -> float:
    """_summary_

    Parameters
    ----------
    y : np.ndarray
        _description_
    rho : np.ndarray
        _description_
    r_s : np.ndarray
        _description_
    w : np.ndarray
        _description_
    op_x : np.ndarray
        _description_
    op_y : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
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
    """_summary_

    Parameters
    ----------
    y : np.ndarray
        _description_
    rho : np.ndarray
        _description_
    r_s : np.ndarray
        _description_
    p : np.ndarray
        _description_
    q_bar : np.ndarray
        _description_
    q_max : np.ndarray
        _description_
    q_min : np.ndarray
        _description_
    w : np.ndarray
        _description_
    op_x : np.ndarray
        _description_
    op_y : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
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
    """_summary_

    Parameters
    ----------
    y : np.ndarray
        _description_
    rho : np.ndarray
        _description_
    r_s : np.ndarray
        _description_
    p : np.ndarray
        _description_
    q_bar : np.ndarray
        _description_
    q_max : np.ndarray
        _description_
    q_min : np.ndarray
        _description_
    w : np.ndarray
        _description_
    op_x : np.ndarray
        _description_
    op_y : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    pass


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
                    vy : np.ndarray) -> float:
    """_summary_

    Parameters
    ----------
    y : np.ndarray
        _description_
    rho : np.ndarray
        _description_
    r_s : np.ndarray
        _description_
    p : np.ndarray
        _description_
    q_bar : np.ndarray
        _description_
    q_max : np.ndarray
        _description_
    q_min : np.ndarray
        _description_
    w : np.ndarray
        _description_
    op_x : np.ndarray
        _description_
    op_y : np.ndarray
        _description_
    vx : np.ndarray
        _description_
    vy : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """
    pass

