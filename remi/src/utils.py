"""Utilities for simulation."""
import os
from typing import Callable
import dill


REMI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')


def load_func(name) -> Callable:
    with open(os.path.join(REMI_DIR, 'functions', name)) as f:
        return dill.load(f)

