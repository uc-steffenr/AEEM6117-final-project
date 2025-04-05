"""Defines framework to run multiple simulations and get metrics."""
from joblib import Parallel, delayed, parallel_backend
from .system import System


def evaluate(args):
    cond, = args
    return cond.run()


class Simulation:
    def __init__(self,
                 conditions : dict,
                 parameters : dict,
                 settings : dict=dict(),
                 n_proc : int=2
                 ) -> None:
        self.n_proc = n_proc
        self.conds = [System(y0, parameters, settings) \
                      for y0 in conditions['y0']]
        self.N = len(self.conds)

    def run_simulations(self) -> dict:
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
                                 backend : str='multiprocessing'
                                 ) -> dict:
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
