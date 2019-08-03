from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from allopy.optimize.uncertainty.result import ConstraintFuncMap, ConstraintMap, Result


class RegretResult(Result):
    def __init__(self, num_assets, num_scenarios):
        super().__init__(num_assets, num_scenarios)
        self.proportions: Optional[np.ndarray] = None
        self.first_level_solutions: Optional[np.ndarray] = None
        self._regret_table: Optional[pd.DataFrame] = None

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
        self.first_level_solutions = np.asarray(solutions)
        self._form_regret_table(obj_funcs)

    def _form_regret_table(self, obj_funcs: List[Callable[[np.ndarray, np.ndarray], float]]):
        # rows -> portfolio, columns -> scenario
        # Last column, "optimal" regret solution in each of the scenario
        objective_fn_values = np.array([
            [f(s) for f in obj_funcs]
            for s in [*self.first_level_solutions, self.solution]
        ])

        self._regret_table = pd.DataFrame([
            objective_fn_values[i, i] - objective_fn_values[:, i]
            for i in range(self.num_scenarios)
        ]).T

    @property
    def regret_table(self):
        assert isinstance(self._regret_table, pd.DataFrame), "Regret information has not been formed yet"

        portfolio_names = [f"{s} Optimal" for s in (*self.scenario_names, "Regret")]
        self._regret_table.index = pd.Index(portfolio_names, name="Portfolio")
        self._regret_table.columns = self.scenario_names

        return self._regret_table
