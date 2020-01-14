from typing import Dict, List, Sized, Union

from .types import Arg1Func, Arg2Func

ConstraintFunc = Union[List[Arg1Func], List[Arg2Func]]


class ConstraintMap:
    def __init__(self, num_scenarios):
        self.n = num_scenarios
        self._equality: Dict[str, ConstraintFunc] = {}
        self._inequality: Dict[str, ConstraintFunc] = {}
        self._m_equality: Dict[str, ConstraintFunc] = {}
        self._m_inequality: Dict[str, ConstraintFunc] = {}

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

    def add_equality_constraints(self, fns: ConstraintFunc):
        self._equality[self._rename(fns[0], self._equality.keys())] = fns

    def add_inequality_constraints(self, fns: ConstraintFunc):
        self._inequality[self._rename(fns[0], self._inequality.keys())] = fns

    def add_matrix_equality_constraints(self, fns: ConstraintFunc):
        self._m_equality[self._rename(fns[0], self._m_equality.keys())] = fns

    def add_matrix_inequality_constraints(self, fns: ConstraintFunc):
        self._m_inequality[self._rename(fns[0], self._m_inequality.keys())] = fns

    @staticmethod
    def _rename(fn: Union[Arg1Func, Arg2Func], names: Sized):
        return f"{fn.__name__}_{len(names) + 1}"
