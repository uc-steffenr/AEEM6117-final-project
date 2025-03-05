"""Defines dynamic system to be propagated."""
import numpy as np
# from scipy.integrate import solve_ivp

from integrate import integrate
from utils import load_dynamics_func


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

    def controls(self, t, y):
        return np.zeros(3)
        # return np.array([2.*np.sin(2.*t), 0., 0.])

    def dynamics(self, t, y, u):        
        yddot = self.dyn(y, u, self.r_s, self.r_t, self.rho, self.m, self.I, self.b).squeeze() 
        return np.hstack((y[4:], yddot))

    def test_event(self, t, y):
        return t >= 5.

    def run(self):
        sol = integrate(
            self.dynamics,
            (0., self.t_dur),
            self.y0,
            self.h,
            self.controls,
            self.test_event
        )

        return sol
