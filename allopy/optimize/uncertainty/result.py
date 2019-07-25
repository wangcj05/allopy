from typing import Callable, Dict, List, Optional

import numpy as np

ConstraintMap = Dict[str, Callable[[np.ndarray], float]]
ConstraintFuncMap = Dict[str, List[Callable[[np.ndarray], float]]]


class Result:
    def __init__(self):
        self.tight_constraint: List[str] = []
        self.violations: List[str] = []
        self.props: Optional[np.ndarray] = None
        self.sol: Optional[np.ndarray] = None

    def update(self,
               eps: float,
               hin: ConstraintFuncMap,
               heq: ConstraintFuncMap,
               min: ConstraintMap,
               meq: ConstraintMap,
               sol: np.ndarray,
               props: Optional[np.ndarray] = None):
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
            for i, f in constraints:
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

        if hasattr(props, "__iter__"):
            self.props = np.asarray(props)
