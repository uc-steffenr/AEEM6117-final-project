"""Utilities for simulation."""
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
