from typing import Optional

import numpy as np

from allopy.optimize.uncertainty.result import ConstraintFuncMap, ConstraintMap, Result


class RegretResult(Result):
    def __init__(self, num_assets, num_scenarios):
        super().__init__(num_assets, num_scenarios)
        self.props: Optional[np.ndarray] = None
        self.first_level_solutions: Optional[np.ndarray] = None

    # noinspection PyMethodOverriding
    def update(self,
               eps: float,
               hin: ConstraintFuncMap,
               heq: ConstraintFuncMap,
               min: ConstraintMap,
               meq: ConstraintMap,
               sol: np.ndarray,
               props: np.ndarray,
               solutions: np.ndarray):
        super().update(eps, hin, heq, min, meq, sol)

        self.props = np.asarray(props)
        self.first_level_solutions = np.asarray(solutions)
