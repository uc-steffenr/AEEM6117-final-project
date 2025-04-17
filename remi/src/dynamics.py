"""Defines dynamics functions."""
import numpy as np
from .utils import load_func


# Load dynamics functions
# ----------------------------------------------------------------------
forward_dynamics = load_func('dynamics.pkl')
inverse_dynamics = load_func('inverse_dynamics.pkl')


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
    pass


def inverse_dynamics(y : np.ndarray,
                     ddths : np.ndarray,
                     r_s : np.ndarray,
                     r_t : np.ndarray,
                     rho : np.ndarray,
                     m : np.ndarray,
                     I : np.ndarray,
                     d : np.ndarray) -> np.ndarray:
    pass
