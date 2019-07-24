from typing import Callable, Dict

import numpy as np

ConstraintMap = Dict[str, Callable]


class Result:
    def __init__(self, hin: ConstraintMap, heq: ConstraintMap, sol: np.ndarray, eps: float):
        self.tight_hin = []
        self.violations = []

        for name, f in hin.items():
            value = abs(f(sol))
            if value <= eps:
                self.tight_hin.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in heq.items():
            if abs(f(sol)) > eps:
                self.violations.append(name)

        self.x = sol
