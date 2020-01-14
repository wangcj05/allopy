from typing import Callable, Dict, List, Sized

import numpy as np

ConstraintFunc = Callable[[np.ndarray], float]


class ConstraintMap:
    def __init__(self, num_scenarios):
        self.n = num_scenarios
        self._equality: Dict[str, List[ConstraintFunc]] = {}
        self._inequality: Dict[str, List[ConstraintFunc]] = {}
        self._m_equality: Dict[str, List[ConstraintFunc]] = {}
        self._m_inequality: Dict[str, List[ConstraintFunc]] = {}

    @property
    def equality(self):
        return self._equality

    @property
    def inequality(self):
        return self._inequality

    @property
    def m_equality(self):
        return self._m_equality

    @property
    def m_inequality(self):
        return self._m_inequality

    def add_equality_constraints(self, fns: List[ConstraintFunc]):
        self._equality[self._rename(fns[0], self._equality.keys())] = fns

    def add_inequality_constraints(self, fns: List[ConstraintFunc]):
        self._inequality[self._rename(fns[0], self._inequality.keys())] = fns

    def add_matrix_equality_constraints(self, fn: ConstraintFunc):
        self._m_equality[self._rename(fn, self._m_equality.keys())] = fn

    def add_matrix_inequality_constraints(self, fn: ConstraintFunc):
        self._m_inequality[self._rename(fn, self._m_inequality.keys())] = fn

    @staticmethod
    def _rename(fn: ConstraintFunc, names: Sized):
        return f"{fn.__name__}_{len(names) + 1}"
