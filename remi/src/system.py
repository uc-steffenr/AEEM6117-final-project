"""Defines dynamic system to be propagated."""
import numpy as np
from typing import Callable

from .integrate import integrate
from .utils import load_dynamics_func


class System:
    def __init__(self,
                 y0 : np.ndarray,
                 parameters : dict,
                 settings : dict=dict()) -> None:
        """Defines the dynamic system to be propagated.

        Parameters
        ----------
        y0 : np.ndarray
            Initial state.
        parameters : dict
            Physical parameters that define the system.
                - r_s - central position of spacecraft
                - r_t - central position of target
                - m - array of mass values for satellite, links, and target
                - rho - array of length values
                - I - array of inertia values
                - b - non-conservative parameters
        settings : dict, optional
            Propagation settings, by default dict().
                - t_dur - duration of propagation
                - step_size - constant step size for propagation
                - tol - tolerance for ending propagation
                - max_tau - maximum torque for controls
        """
        self.y0 = y0

        self.r_s = np.reshape(parameters['r_s'], (1, 2))
        self.r_t = np.reshape(parameters['r_t'], (1, 2))
        self.m = parameters['m']
        self.rho = parameters['rho']
        self.I = parameters['I']
        self.b = parameters['b']

        self.t_dur = settings.get('t_dur', 15.)
        self.h = settings.get('step_size', 0.01)
        self.tol = settings.get('tol', 0.01)
        self.max_tau = np.array(settings.get('max_tau', (10., 10., 10.)))

        self.dyn = load_dynamics_func()
        self.event = None
        self.F = None

    def set_controller(self, F : Callable[[float, np.ndarray], np.ndarray]) -> None:
        """Defines controller for system.

        Parameters
        ----------
        F : Callable[[float, np.ndarray], np.ndarray]
            Control function. Takes time and state as input, returns 
            control array.
        """
        self.F = F

    def set_event(self, E : Callable[[float, np.ndarray, float], bool]) -> None:
        """Even to end propagation.

        Parameters
        ----------
        E : Callable[[float, np.ndarray, float], bool]
            Terminating event. Takes time, state, and tolerance as input,
            returns boolean where True means event occurred.
        """
        self.event = E

    # TODO: probably need target information accessible here
    def controls(self, t : float, y : np.ndarray) -> np.ndarray:
        """Control function used for propagation.

        Parameters
        ----------
        t : float
            Current time.
        y : np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Control torques.
        """
        if self.F is None:
            return np.zeros(3)

        return np.clip(self.F(t, y), -self.max_tau, self.max_tau)

    def dynamics(self,
                 t : float,
                 y : np.ndarray,
                 u : np.ndarray) -> np.ndarray:
        """Dynamics function to be propagated.

        Parameters
        ----------
        t : float
            Current time.
        y : np.ndarray
            Current state.
        u : np.ndarray
            Control input.

        Returns
        -------
        np.ndarray
            Time derivatives of states.
        """
        yddot = self.dyn(y,
                         u,
                         self.r_s,
                         self.r_t,
                         self.rho,
                         self.m,
                         self.I,
                         self.b
                         ).squeeze() 
        return np.hstack((y[4:], yddot))

    def run(self):
        """Method to run propagation. Returns Solution class.
        """
        def event(t, y):
            if self.event is not None:
                return self.event(t, y, self.tol)

            return False

        sol = integrate(
            self.dynamics,
            (0., self.t_dur),
            self.y0,
            self.h,
            self.controls,
            event
        )

        return sol
