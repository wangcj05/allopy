from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from allopy import get_option
from ._modelbuilder import ModelBuilder


@dataclass
class RegretOptimizerSolution:
    regret_optimal: np.ndarray
    scenario_optimal: np.ndarray
    proportions: Optional[np.ndarray] = None


@dataclass
class RegretResult:
    num_assets: int
    num_scenarios: int
    tight_constraint: List[str]
    violations: List[str]
    solution: Optional[np.ndarray]
    _assets: List[str]
    _scenarios: List[str]
    proportions: Optional[np.ndarray]
    scenario_solutions: Optional[np.ndarray]

    def __init__(self,
                 mb: ModelBuilder,
                 solution: np.ndarray,
                 scenario_solutions: np.ndarray,
                 proportions: Optional[np.ndarray],
                 eps: float = get_option("EPS.CONSTRAINT")):
        self.num_assets = mb.num_assets
        self.num_scenarios = mb.num_scenarios
        self.solution = np.asarray(solution)
        self.proportions = np.asarray(proportions)
        self.scenario_solutions = np.asarray(scenario_solutions)

        self._assets = [f"Asset_{i + 1}" for i in range(mb.num_assets)]
        self._scenarios: List[str] = [f"Scenario_{i + 1}" for i in range(mb.num_scenarios)]

        self.tight_constraint: List[str] = []
        self.violations: List[str] = []

        self._check_matrix_constraints(mb.constraints, self.solution, eps)
        self._check_functional_constraints(mb.constraints, self.solution, eps)

    @property
    def assets(self):
        return self._assets

    @assets.setter
    def assets(self, value: List[str]):
        error = f"asset_names must be a list with {self.num_assets} unique names"
        assert hasattr(value, "__iter__"), error

        value = list(set([str(i) for i in value]))
        assert len(value) == self.num_assets, error

        self._assets = value

    @property
    def scenarios(self):
        return self._scenarios

    @scenarios.setter
    def scenarios(self, value: List[str]):
        error = f"scenario_names must be a list with {self.num_scenarios} unique names"
        assert hasattr(value, "__iter__"), error

        value = list(set([str(i) for i in value]))
        assert len(value) == self.num_scenarios, error

        self._scenarios = value

    def _check_functional_constraints(self, constraints, solution, eps):
        for name, cstr in constraints.inequality.items():
            for i, f in enumerate(cstr):
                value = f(solution)
                if np.isclose(value, 0, atol=eps):
                    self.tight_constraint.append(f"{name}-{i}")
                elif value > eps:
                    self.violations.append(f"{name}-{i}")

        for name, cstr in constraints.equality.items():
            for i, f in enumerate(cstr):
                if abs(f(solution)) > eps:
                    self.violations.append(f"{name}-{i}")

    def _check_matrix_constraints(self, constraints, solution, eps):
        for name, f in constraints.m_equality.items():
            value = f(solution)
            if abs(value) <= eps:
                self.tight_constraint.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in constraints.m_inequality.items():
            if abs(f(solution)) > eps:
                self.violations.append(name)
