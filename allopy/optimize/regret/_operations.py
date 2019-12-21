from typing import Callable, List, Optional, Union

import nlopt as nl
import numpy as np

from allopy.optimize.base import BaseOptimizer
from allopy.types import OptArray
from ._modelbuilder import ModelBuilder
from .result import RegretOptimizerSolution, RegretResult


class OptimizationOperation:
    def __init__(self, builder: ModelBuilder, prob: np.ndarray, max_attempts: int, verbose: bool):
        self.builder = builder
        self.prob = prob
        self.solution: Optional[RegretOptimizerSolution] = None
        self.result: Optional[RegretResult] = None
        self.max_attempts = max_attempts
        self.verbose = verbose

    def optimize(self,
                 x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                 x0_prop: OptArray = None,
                 initial_solution: Optional[str] = "random",
                 approx=True,
                 dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                 random_state: Optional[int] = None):
        if isinstance(random_state, int) and isinstance(initial_solution, str) and initial_solution.lower() == "random":
            np.random.seed(random_state)

        self._validate_dist_func(dist_func)
        x0_first_level = self._validate_first_level_solution(x0_first_level)
        mb = self.builder
        num_scenarios, num_assets = mb.num_scenarios, mb.num_assets

        # optimal solution to each scenario. Each row represents a single scenario and
        # each column represents an asset class
        solutions = np.array([
            self._optimize(mb(i), x0_first_level[i], initial_solution)
            for i in range(num_scenarios)
        ])

        if np.isnan(solutions).any():
            props = np.repeat(np.nan, num_scenarios) if approx else None
            weights = np.repeat(np.nan, num_assets)
        elif approx:
            props, weights = self._optimize_approx(x0_prop, solutions, dist_func, initial_solution)
        else:
            props, weights = self._optimize_actual(x0_prop, solutions, dist_func, initial_solution)

        self.solution = RegretOptimizerSolution(weights, solutions, props)
        self.result = RegretResult(self.builder, weights, solutions, props, mb.c_eps)

        return self

    def _optimize(self, model: BaseOptimizer, x0: OptArray = None, initial_solution=None):
        """Helper method to run the model"""

        for _ in range(self.max_attempts):
            try:
                w = model.optimize(x0, initial_solution=initial_solution)
                if w is None or np.isnan(w).any():
                    if initial_solution is None:
                        initial_solution = "random"
                    x0 = None
                else:
                    return w

            except (nl.RoundoffLimited, RuntimeError):
                if initial_solution is None:
                    initial_solution = "random"
                x0 = None
        else:
            if self.verbose:
                print('No solution was found for the given problem. Check the summary() for more information')
            return np.repeat(np.nan, self.builder.num_assets)

    def _optimize_actual(self,
                         x0: OptArray,
                         solutions: np.ndarray,
                         dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc],
                         initial_solution: Optional[str] = None):
        """
        Runs the second step (regret minimization) using the actual weights as the decision variable

        Parameters
        ----------
        x0
            Initial solution. If provided, this must be the final portfolio weights

        solutions
            Matrix of solution where the rows represents the weights for the scenario and the
            columns represent the asset classes

        dist_func: Callable
            Distance function to scale the objective function

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied. See notes on
            Initial Solution for more information
        """
        builder = self.builder
        f_values = np.array(f(s) for f, s in zip(builder.obj_funcs, solutions))

        def regret(w):
            curr_f_values = np.array([f(w) for f in builder.obj_funcs])
            cost = dist_func(f_values - curr_f_values)
            return 100 * sum(self.prob * cost)

        model = BaseOptimizer(builder.num_assets)
        model.set_min_objective(regret)
        model.set_bounds(builder.lower_bounds, builder.upper_bounds)

        constraints = builder.constraints
        for cs, set_constraint in [(constraints.m_equality.values(), model.add_equality_matrix_constraint),
                                   (constraints.m_inequality.values(), model.add_inequality_matrix_constraint)]:
            for c in cs:
                set_constraint(c, builder.c_eps)

        return None, self._optimize(model,
                                    self.prob @ solutions if x0 is None else x0,
                                    initial_solution)

    def _optimize_approx(self,
                         x0: OptArray,
                         solutions: np.ndarray,
                         dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc],
                         initial_solution: Optional[str] = None):
        """
        Runs the second step (regret minimization) where the decision variable

        Parameters
        ----------
        x0
            Initial solution. If provided, this must be the proportion of each scenario's contribution to
            the final portfolio weights

        solutions
            Matrix of solution where the rows represents the weights for the scenario and the
            columns represent the asset classes

        dist_func: Callable
            Distance function to scale the objective function

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied. See notes on
            Initial Solution for more information
        """
        # weighted function values for each scenario
        builder = self.builder
        f_values: np.ndarray = np.array([f(s) for f, s in zip(builder.obj_funcs, solutions)])

        def regret(p):
            cost = f_values - np.array([f(p @ solutions) for f in builder.obj_funcs])
            cost = dist_func(cost)
            return 100 * sum(self.prob * cost)

        model = BaseOptimizer(builder.num_scenarios)
        model.set_min_objective(regret)

        model.set_bounds(0, 1)
        proportions = self._optimize(model,
                                     self.prob if x0 is None else x0,
                                     initial_solution)
        return proportions, proportions @ solutions

    def _validate_dist_func(self, dist_func):
        assert callable(dist_func), "dist_func must be a callable function"

        assert isinstance(dist_func(np.random.uniform(size=self.builder.num_assets)), np.ndarray), \
            "dist_func must map a vector to a vector"

    def _validate_first_level_solution(self, x0_first_level: Optional[Union[List[OptArray], np.ndarray]]):
        if x0_first_level is None:
            return [None] * self.builder.num_scenarios

        assert len(x0_first_level) == self.builder.num_scenarios, \
            "Initial first level solution data must match number of scenarios"

        return x0_first_level
