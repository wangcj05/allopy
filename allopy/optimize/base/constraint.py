from typing import Callable, Dict, Sized

import numpy as np

ConstraintFunc = Callable[[np.ndarray], float]


class ConstraintMap:
    def __init__(self):
        self._equality: Dict[str, ConstraintFunc] = {}
        self._inequality: Dict[str, ConstraintFunc] = {}

    def add_equality_constraint(self, f: ConstraintFunc):
        self._equality[self._rename(f, self._equality.keys())] = f

    def add_inequality_constraint(self, f: ConstraintFunc):
        self._inequality[self._rename(f, self._inequality.keys())] = f

    @property
    def equality(self):
        return self._equality

    @property
    def inequality(self):
        return self._inequality

    @staticmethod
    def _rename(fn: ConstraintFunc, names: Sized):
        return f"{fn.__name__}_{len(names) + 1}"
