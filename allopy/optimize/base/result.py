import numpy as np

from .constraint import ConstraintMap


class Result:
    def __init__(self, constraints: ConstraintMap, sol: np.ndarray, eps: float):
        self.tight_hin = []
        self.violations = []

        for name, f in constraints.inequality.items():
            value = abs(f(sol))
            if value <= eps:
                self.tight_hin.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in constraints.equality.items():
            if abs(f(sol)) > eps:
                self.violations.append(name)

        self.x = sol
