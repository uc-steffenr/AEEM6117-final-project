"""Defines dynamic system to be propagated."""
import numpy as np
from scipy.integrate import solve_ivp
from utils import load_dynamics_func


class System:
    def __init__(self, y0, parameters, settings=dict()):
        self.y0 = y0

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
        # return np.array([2.*np.sin(2.*t), 0., np.cos(2.*t)])

    def dynamics(self, t, y):
        # controls goes here
        u = self.controls(t, y)
        
        yddot = self.dyn(y, u, self.rho, self.m, self.I, self.b).squeeze() 
        return np.hstack((y[3:], yddot))

    def run(self, **solve_ivp_kwargs):
        sol = solve_ivp(
            self.dynamics,
            (0., self.t_dur),
            self.y0,
            t_eval=np.arange(0., self.t_dur+self.h, self.h),
            **solve_ivp_kwargs
        )

        # Method to extract controls later here
        vfunc = np.vectorize(self.controls,
                            #  otypes=[np.ndarray],
                             excluded=[1],
                             cache=True,
                             signature="()->(n)")
        us = vfunc(sol.t, sol.y.T)

        return sol.t, sol.y, us
