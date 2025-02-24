"""Utilities for Sim."""
import numpy as np
import dill

def load_position_func():
    with open('functions/position.pkl', 'rb') as f:
        return dill.load(f)

def load_velocity_func():
    with open('functions/velocity.pkl', 'rb') as f:
        return dill.load(f)

def load_dynamics_func():
    with open('functions/dynamics.pkl', 'rb') as f:
        return dill.load(f)


if __name__ == '__main__':
    rho = np.array([0.5, 0.5, 0.5])
    m = np.array([250., 25., 25.])
    I = np.array([25., 2.5, 2.5])
    b = np.zeros(3)

    y = np.zeros(6)
    u = np.zeros(3)
    u[1] = np.deg2rad(30.)

    pos = load_position_func()
    vel = load_velocity_func()
    dyn = load_dynamics_func()

    r_s, r_1, r_2 = pos(y, rho)
    v_s, v_1, v_2 = vel(y, rho)
    qddot = dyn(y, u, rho, m, I, b)

    print('Position Test')
    print('-'*40)
    print(f'r_s = {r_s}')
    print(f'r_1 = {r_1}')
    print(f'r_2 = {r_2}')
    print()

    print('Velocity Test')
    print('-'*40)
    print(f'v_s = {v_s}')
    print(f'v_1 = {v_1}')
    print(f'v_2 = {v_2}')
    print()

    print('Dynamics Test')
    print('-'*40)
    print(f'qddot = {qddot}')
