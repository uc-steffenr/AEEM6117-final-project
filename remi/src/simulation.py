"""Defines framework to run multiple simulations and get metrics."""
import numpy as np
from typing import Callable
from joblib import Parallel, delayed, parallel_backend

from .system import System


def evaluate(args : tuple[System]):
    cond, = args
    return cond.run()


class Simulation:
    def __init__(self,
                 conditions : list[np.ndarray],
                 parameters : dict,
                 settings : dict=dict(),
                 n_proc : int=2
                 ) -> None:
        """Class for multiple systems to be run.

        Parameters
        ----------
        conditions : list[np.ndarray]
            Initial conditions for each system.
        parameters : dict
            Physical parameters.
        settings : dict, optional
            Simulation settings, by default dict().
        n_proc : int, optional
            Number of processors for running in parallel, by default 2.
        """
        self.n_proc = n_proc
        self.conds = [System(y0, parameters, settings) \
                      for y0 in conditions]
        self.N = len(self.conds)


    def set_controller(self, F : Callable[[float, np.ndarray], np.ndarray]):
        """Sets controller for every condition.

        Parameters
        ----------
        F : Callable[[float, np.ndarray], np.ndarray]
            Control function. Takes time and state as input, returns
            controll array.
        """
        for i in range(self.N):
            self.conds[i].set_controller(F)


    def set_event(self, E : Callable[[float, np.ndarray, float], bool]):
        """Sets terminating event for every condition.

        Parameters
        ----------
        E : Callable[[float, np.ndarray, float], bool]
            Terminating event. Takes time, state, and tolerance as input,
            returns boolean where True means event occurred.
        """
        for i in range(self.N):
            self.conds[i].set_event(E)


    def run_simulations(self) -> dict:
        """Method to run simulations in serial.

        Returns
        -------
        dict
            Dictionary of run metrics.
        """
        metrics = dict(ts=[None]*self.N,
                       ys=[None]*self.N,
                       us=[None]*self.N,
                       statuses=[None]*self.N)

        for i, cond in enumerate(self.conds):
            sol = cond.run()
            metrics['ts'][i] = sol.t
            metrics['ys'][i] = sol.y
            metrics['us'][i] = sol.u
            metrics['statuses'][i] = sol.status

        return metrics


    def run_parallel_simulations(self,
                                 backend : str='loky'
                                 ) -> dict:
        """Method to run simulations in parallel.

        Parameters
        ----------
        backend : str, optional
            Parallelization method, by default 'multiprocessing'.

        Returns
        -------
        dict
            Dictionary of run matrics.
        """
        metrics = dict(ts=[None]*self.N,
                       ys=[None]*self.N,
                       us=[None]*self.N,
                       statuses=[None]*self.N)

        args = [(cond,) for cond in self.conds]

        with parallel_backend(backend, n_jobs=self.n_proc):
            results = Parallel()(delayed(evaluate)(arg) for arg in args)

        for i, sol in enumerate(results):
            metrics['ts'][i] = sol.t
            metrics['ys'][i] = sol.y
            metrics['us'][i] = sol.u
            metrics['statuses'][i] = sol.status

        return metrics
