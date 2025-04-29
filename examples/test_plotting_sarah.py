import os
import numpy as np
from remi.system import System
from remi.visualize import plot_states, plot_controls, animate
from remi.plots import Plots

# Physical Parameters
r_s = np.array([-1.5, 0.0])
r_t = np.array([1.5, 0.0])
rho = np.array([0.5, 0.5, 0.5, 0.5])
m = np.array([250.0, 25.0, 25.0, 180.0])
I = np.array([25.0, 2.5, 2.5, 18.0])
d = np.zeros(4)

# Simulation Settings
t_dur = 10.0
step_size = 0.01
tol = 0.01
max_tau = (10.0, 10.0, 10.0, 0.0)

# Initial Conditions
y0 = np.array(
    [0.0, 0.0, 0.0, 0.0, np.deg2rad(5.0), np.deg2rad(10.0), 0.0, np.deg2rad(-5.0)]
)


# Put parameters and settings in dict
parameters = dict(r_s=r_s, r_t=r_t, rho=rho, m=m, I=I, d=d)

settings = dict(t_dur=t_dur, step_size=step_size, tol=tol, max_tau=max_tau)

# Define system
sys = System(y0, parameters, settings)


# Define event to end propagation
# here, we want to end propagation once the time is more than 5 seconds
def event(t, y, tol):
    return t >= 5.0


# Define controls of simulation
def controls(t, y):
    return np.zeros(4)


# Set event and controller
sys.set_controller(controls)
sys.set_event(event)

# Right the simulation
sol = sys.run()

print(f"Sim status: {sol.status}")

dp = Plots()
for t, y, u in zip(sol.t, sol.y, sol.u):
    dp.update(t, y, u)
