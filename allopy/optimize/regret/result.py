from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from allopy import get_option

ConstraintMap = Dict[str, Callable[[np.ndarray], float]]
ConstraintFuncMap = Dict[str, List[Callable[[np.ndarray], float]]]


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
    _asset_names: List[str]
    _scenario_names: List[str]
    proportions: Optional[np.ndarray]
    scenario_solutions: Optional[np.ndarray]

    def __init__(self,
                 num_assets: int,
                 num_scenarios: int,
                 solution: np.ndarray,
                 scenario_solutions: np.ndarray,
                 proportions: Optional[np.ndarray],
                 hin: ConstraintFuncMap,
                 heq: ConstraintFuncMap,
                 min: ConstraintMap,
                 meq: ConstraintMap,
                 eps: float = get_option("EPS.CONSTRAINT")):
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios
        self.solution = np.asarray(solution)
        self.proportions = np.asarray(proportions)
        self.scenario_solutions = np.asarray(scenario_solutions)

        self._asset_names = [f"Asset_{i + 1}" for i in range(num_assets)]
        self._scenario_names: List[str] = [f"Scenario_{i + 1}" for i in range(num_scenarios)]

        self.tight_constraint: List[str] = []
        self.violations: List[str] = []

        for name, f in min.items():
            value = f(solution)
            if abs(value) <= eps:
                self.tight_constraint.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in meq.items():
            if abs(f(solution)) > eps:
                self.violations.append(name)

        for name, constraints in hin.items():
            for i, f in enumerate(constraints):
                value = f(solution)
                if np.isclose(value, 0, atol=eps):
                    self.tight_constraint.append(f"{name}-{i}")
                elif value > eps:
                    self.violations.append(f"{name}-{i}")

        for name, constraints in heq.items():
            for i, f in enumerate(constraints):
                if abs(f(solution)) > eps:
                    self.violations.append(f"{name}-{i}")

    @property
    def asset_names(self):
        return self._asset_names

    @asset_names.setter
    def asset_names(self, value: List[str]):
        error = f"asset_names must be a list with {self.num_assets} unique names"
        assert hasattr(value, "__iter__"), error

        value = list(set([str(i) for i in value]))
        assert len(value) == self.num_assets, error

        self._asset_names = value

    @property
    def scenario_names(self):
        return self._scenario_names

    @scenario_names.setter
    def scenario_names(self, value: List[str]):
        error = f"scenario_names must be a list with {self.num_scenarios} unique names"
        assert hasattr(value, "__iter__"), error

        value = list(set([str(i) for i in value]))
        assert len(value) == self.num_scenarios, error

        self._scenario_names = value
