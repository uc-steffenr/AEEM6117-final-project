"""Utilities for simulation."""
import os
import dill

_pth = os.path.join('/', *__file__.split('/')[:-1])

def load_position_func():
    with open(os.path.join(_pth, 'functions', 'position.pkl'), 'rb') as f:
        return dill.load(f)

def load_velocity_func():
    with open(os.path.join(_pth, 'functions', 'velocity.pkl'), 'rb') as f:
        return dill.load(f)

def load_dynamics_func():
    with open(os.path.join(_pth, 'functions', 'dynamics.pkl'), 'rb') as f:
        return dill.load(f)
