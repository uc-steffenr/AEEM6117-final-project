"""Defines dynamic system to be propagated."""
import numpy as np

from src.integrate import integrate
from src.utils import load_dynamics_func


class System:
    def __init__(self, y0, parameters, settings=dict()):
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

    def set_controller(self, F):
        self.F = F

    def set_event(self, E):
        self.event = E

    def controls(self, t, y):
        if self.F is None:
            return np.zeros(3)

        return self.F(t, y)

    def dynamics(self, t, y, u):        
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
        sol = integrate(
            self.dynamics,
            (0., self.t_dur),
            self.y0,
            self.h,
            self.controls,
            self.event
        )

        return sol
