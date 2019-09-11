from typing import Callable, List, Optional

import numpy as np

from allopy.optimize.uncertainty.result import ConstraintFuncMap, ConstraintMap, Result


class RegretResult(Result):
    def __init__(self, num_assets, num_scenarios):
        super().__init__(num_assets, num_scenarios)
        self.proportions: Optional[np.ndarray] = None
        self.scenario_solutions: Optional[np.ndarray] = None

    # noinspection PyMethodOverriding
    def update(self,
               eps: float,
               hin: ConstraintFuncMap,
               heq: ConstraintFuncMap,
               min: ConstraintMap,
               meq: ConstraintMap,
               sol: np.ndarray,
               proportions: np.ndarray,
               solutions: np.ndarray,
               obj_funcs: List[Callable[[np.ndarray, np.ndarray], float]]):
        super().update(eps, hin, heq, min, meq, sol)

        self.proportions = np.asarray(proportions)
        self.scenario_solutions = np.asarray(solutions)
