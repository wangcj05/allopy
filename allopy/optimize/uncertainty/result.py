from typing import Callable, Dict, List, Optional

import numpy as np

ConstraintMap = Dict[str, Callable[[np.ndarray], float]]
ConstraintFuncMap = Dict[str, List[Callable[[np.ndarray], float]]]


class Result:
    def __init__(self, num_assets: int, num_scenarios: int):
        self.num_assets = num_assets
        self.num_scenarios = num_scenarios

        self.tight_constraint: List[str] = []
        self.violations: List[str] = []
        self.sol: Optional[np.ndarray] = None

        self._asset_names = [f"Asset_{i + 1}" for i in range(num_assets)]
        self._scenario_names: List[str] = [f"Scenario_{i + 1}" for i in range(num_scenarios)]

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

    def update(self,
               eps: float,
               hin: ConstraintFuncMap,
               heq: ConstraintFuncMap,
               min: ConstraintMap,
               meq: ConstraintMap,
               sol: np.ndarray,
               *args,
               **kwargs):
        self.tight_constraint = []
        self.violations = []

        for name, f in min.items():
            value = f(sol)
            if abs(value) <= eps:
                self.tight_constraint.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in meq.items():
            if abs(f(sol)) > eps:
                self.violations.append(name)

        for name, constraints in hin.items():
            for i, f in enumerate(constraints):
                value = f(sol)
                if np.isclose(value, 0, atol=eps):
                    self.tight_constraint.append(f"{name}-{i}")
                elif value > eps:
                    self.violations.append(f"{name}-{i}")

        for name, constraints in heq.items():
            for i, f in enumerate(constraints):
                if abs(f(sol)) > eps:
                    self.violations.append(f"{name}-{i}")

        self.sol = sol
