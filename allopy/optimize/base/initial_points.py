from typing import Callable, List

import numpy as np


class InitialPointGenerator:
    def __init__(self, n: int, lb: np.ndarray, ub: np.ndarray):
        self.n = n
        self.lb = lb
        self.ub = ub

    def random_starting_points(self, random_state: int = None):
        if random_state is not None:
            np.random.seed(random_state)

        return np.random.uniform(self.lb, self.ub)

    def min_constraint(self, eq_cons: List[Callable[[np.ndarray], float]],
                       ineq_cons: List[Callable[[np.ndarray], float]]):
        from allopy.optimize.base import BaseOptimizer

        model = BaseOptimizer(self.n)
        model.set_bounds(self.lb, self.ub)

        def obj_fun(w):
            return (sum([1e5 * f(w) ** 2 for f in eq_cons]) + sum([f(w) ** 2 for f in ineq_cons])) ** 0.5

        model.set_min_objective(obj_fun)
        return model.optimize(self.random_starting_points())
