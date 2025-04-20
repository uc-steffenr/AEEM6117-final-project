"""Defines integration method to be used."""
import numpy as np


class Solution:
    def __init__(self):
        self.t = None
        self.y = None
        self.u = None
        self.status = None


def RK4(func, t, y, h, args):
    k1 = h * func(t, y, *args)
    k2 = h * func(t + 0.5*h, y + 0.5*k1, *args)
    k3 = h * func(t + 0.5*h, y + 0.5*k2, *args)
    k4 = h * func(t + h, y + k3, *args)

    return y + (1./6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def integrate(plant, t_span, y0, step_size, F, event=None):
    ts = np.arange(t_span[0], t_span[1], step_size)
    n = len(ts)

    sol = Solution()
    sol.t = ts
    sol.y = np.zeros((n, y0.size))
    sol.u = np.zeros((n-1, 4))
    sol.y[0, :] = y0

    # Main integration loop
    y = y0
    for i in range(1, n):
        # try:
        u = F(ts[i], y)
        y = RK4(plant, ts[i], y, step_size, (u,))
        # Integration error
        # except Exception as e:
        #     print(f'WARNING: {e}')
        #     sol.y = sol.y[:i, :]
        #     sol.t = ts[:i]
        #     sol.u = sol.u[:i, :]
        #     sol.status = -1
        #     return sol

        sol.y[i, :] = y
        sol.u[i-1, :] = u

        if event is not None and event(ts[i], y):
            sol.y = sol.y[:i+1, :]
            sol.t = ts[:i+1]
            sol.u = sol.u[:i, :]
            sol.status = 1
            return sol

    sol.status = 0
    return sol
