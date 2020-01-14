import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional

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
    def __init__(self,
                 mb: ModelBuilder,
                 solution: np.ndarray,
                 scenario_solutions: np.ndarray,
                 proportions: Optional[np.ndarray],
                 dist_func: Callable[[np.ndarray], np.ndarray],
                 probability: np.ndarray,
                 eps: float = get_option("EPS.CONSTRAINT")):
        self.num_assets = mb.num_assets
        self.num_scenarios = mb.num_scenarios
        self.solution = np.asarray(solution)
        self.proportions = np.asarray(proportions) if proportions is not None else None
        self.scenario_solutions = np.asarray(scenario_solutions)
        self.probability = probability

        self._assets = [f"Asset_{i + 1}" for i in range(mb.num_assets)]
        self._scenarios = [f"Scenario_{i + 1}" for i in range(mb.num_scenarios)]

        self.tight_constraint: List[str] = []
        self.violations: List[str] = []

        self._check_matrix_constraints(mb.constraints, eps)
        self._check_functional_constraints(mb.constraints, eps)

        self.scenario_objective_values = self._derive_scenario_objective_values(mb)
        self.regret_value = self._derive_regret_value(mb, dist_func)
        self.constraint_values = self._derive_constraint_values(mb)

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

    def _check_functional_constraints(self, constraints, eps):
        for name, cstr in constraints.inequality.items():
            for i, f in enumerate(cstr):
                value = f(self.solution)
                if np.isclose(value, 0, atol=eps):
                    self.tight_constraint.append(f"{name}-{i}")
                elif value > eps:
                    self.violations.append(f"{name}-{i}")

        for name, cstr in constraints.equality.items():
            for i, f in enumerate(cstr):
                if abs(f(self.solution)) > eps:
                    self.violations.append(f"{name}-{i}")

    def _check_matrix_constraints(self, constraints, eps):
        for name, fns in constraints.m_equality.items():
            for f in fns:
                value = f(self.solution)
                if abs(value) <= eps:
                    self.tight_constraint.append(name)
                elif value > eps:
                    self.violations.append(name)

        for name, fns in constraints.m_inequality.items():
            for f in fns:
                if abs(f(self.solution)) > eps:
                    self.violations.append(name)

    def _derive_scenario_objective_values(self, mb: ModelBuilder):
        values = []
        for f, s in zip(mb.obj_funcs, self.scenario_solutions):
            if len(inspect.signature(f).parameters) == 1:
                v = f(s)
            else:  # number of parameters can only be 2 in this case
                grad = np.ones((self.num_assets, self.num_assets))  # filler gradient, not necessary
                v = f(s, grad)
            values.append(v / get_option("F.SCALE"))

        return np.array(values)

    def _derive_regret_value(self, mb: ModelBuilder, dist_func: Callable[[np.ndarray], np.ndarray]) -> float:
        f_values = np.array([f(s) for f, s in zip(mb.obj_funcs, self.scenario_solutions)])
        curr_f_values = np.array([f(self.solution) for f in mb.obj_funcs])

        cost = dist_func(f_values - curr_f_values) / get_option("F.SCALE")
        return sum(self.probability * cost)

    def _derive_constraint_values(self, mb: ModelBuilder):
        constraints = []
        for eq, constraint_map in [("<=", mb.constraints.m_inequality),
                                   ("<=", mb.constraints.inequality),
                                   ("=", mb.constraints.m_equality),
                                   ("=", mb.constraints.equality)]:
            for name, fns in constraint_map.items():
                for f, s in zip(fns, self.scenarios):
                    if len(inspect.signature(f).parameters) == 1:
                        v = f(self.solution)
                    else:  # number of parameters can only be 2 in this case
                        grad = np.ones((self.num_assets, self.num_assets))  # filler gradient, not necessary
                        v = f(self.solution, grad)

                    constraints.append({
                        "Name": name,
                        "Scenario": s,
                        "Equality": eq,
                        "Value": v / get_option("C.SCALE")
                    })

        return constraints
