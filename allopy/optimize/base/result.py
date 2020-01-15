import inspect
from typing import Callable, List, Optional

import numpy as np

from allopy import get_option
from .constraint import ConstraintMap


class Result:
    def __init__(self):
        self.obj_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
        self.tight_hin: List[str] = []
        self.violations: List[str] = []
        self.constraint_values = []

        self._x: Optional[np.ndarray] = None

    @property
    def obj_value(self):
        assert self.obj_func is not None
        if len(inspect.signature(self.obj_func).parameters) == 1:
            return self.obj_func(self.x)

        return self.obj_func(self.x, np.ones((len(self.x), len(self.x))))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, solution: np.ndarray):
        self._x = solution

    def set_constraints(self, constraints: ConstraintMap, eps: float):
        assert self.x is not None

        for name, f in constraints.inequality.items():
            value = abs(f(self.x))
            if value <= eps:
                self.tight_hin.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in constraints.equality.items():
            if abs(f(self.x)) > eps:
                self.violations.append(name)

        for eq, c_map in [("<=", constraints.inequality),
                          ("=", constraints.equality)]:
            for name, f in c_map.items():
                if len(inspect.signature(f).parameters) == 1:
                    v = f(self.x) / get_option("C.SCALE")
                else:  # at most 2 parameters
                    v = f(self.x, np.ones((len(self.x), len(self.x)))) / get_option("C.SCALE")

                self.constraint_values.append({
                    "Name": name,
                    "Equality": eq,
                    "Value": v
                })
