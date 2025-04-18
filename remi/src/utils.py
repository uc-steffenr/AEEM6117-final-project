"""Utilities for simulation."""

import os
from typing import Callable
import cloudpickle


REMI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")


def load_func(name) -> Callable:
    with open(os.path.join(REMI_DIR, "functions", name), "rb") as f:
        return cloudpickle.load(f)
