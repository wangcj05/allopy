from typing import Callable

import nlopt as nl
import numpy as np

from allopy import BaseOptimizer
from allopy.types import OptArray
from .funcs import *
from ..algorithms import LD_SLSQP
from ..uncertainty import DiscreteUncertaintyOptimizer

__all__ = ['RegretOptimizer']


class RegretOptimizer(DiscreteUncertaintyOptimizer):
    def __init__(self,
                 n: int,
                 algorithm=LD_SLSQP,
                 rebalance=False,
                 *args,
                 **kwargs):
        """
        The RegretOptimizer houses several common pre-specified optimization regimes for scenario based optimization.

        Notes
        -----


        Parameters
        ----------
        n: int
            number of assets

        algorithm: {int, string}
            The algorithm used for optimization. Default is Sequential Least Squares Programming

        rebalance: bool, optional
            Whether the weights are rebalanced in every time instance. Defaults to False

        args:
            other arguments to pass to the :class:`BaseOptimizer`

        kwargs:
            other keyword arguments to pass into :class:`OptData` (if you passed in a numpy array for `data`) or into
            the :class:`BaseOptimizer`

        See Also
        --------
        :class:`DiscreteUncertaintyOptimizer`: Discrete Uncertainty Optimizer
        """
        super().__init__(n, algorithm, *args, **kwargs)
        self._rebalance = rebalance

        self._max_attempts = 0
        self.max_attempts = kwargs.get('max_attempts', 100)

    def optimize(self,
                 x0: OptArray = None,
                 random_start=False,
                 approx=True,
                 dist_func: Callable[[np.ndarray], np.ndarray] = lambda x: x ** 2):
        # optimal solution to each scenario. Each row represents a single scenario and
        # each column represents an asset class
        solutions = np.array([
            self._optimize(
                self._setup_optimization_model(i),
                x0,
                random_start
            ) for i in range(self._num_scenarios)
        ])

        if approx:
            prop, weights = self._optimize_approx(solutions, dist_func)
        else:
            weights = self._optimize_actual(solutions, dist_func)

        return weights

    def _optimize(self, model: BaseOptimizer, x0: OptArray = None, random_start=False):
        for _ in range(self.max_attempts):
            try:
                w = model.optimize(x0, random_start=random_start)
                if w is not None:
                    return w

            except (nl.RoundoffLimited, RuntimeError):
                x0 = np.random.uniform(self.lower_bounds, self.upper_bounds)
        else:
            if self._verbose:
                print('No solution was found for the given problem. Check the summary() for more information')
            return np.repeat(np.nan, self._num_assets)

    def _optimize_actual(self, solutions: np.ndarray, dist_func: Callable):
        f_values = np.array(self._obj_fun[i](s) for i, s in enumerate(solutions))

        def regret(w):
            curr_f_values = np.array([f(w) for f in self._obj_fun])
            cost = f_values - curr_f_values
            if callable(dist_func):
                cost = np.asarray(dist_func(cost))
            return sum(self.prob * cost)

        model = BaseOptimizer(self._num_scenarios)
        model.set_min_objective(regret)
        model.set_bounds(self.lower_bounds, self.upper_bounds)

        for constraints, set_constraint in [(self._meq.values(), model.add_equality_matrix_constraint),
                                            (self._min.values(), model.add_inequality_matrix_constraint)]:
            for c in constraints:
                set_constraint(c, self._c_eps)

        return self._optimize(model)

    def _optimize_approx(self, solutions: np.ndarray, dist_func: Callable):

        # weighted function values for each scenario
        f_values: np.ndarray = np.array([self._obj_fun[i](s) for i, s in enumerate(solutions)])

        def regret(p):
            curr_f_values = np.array([self._obj_fun[i](p @ s) for i, s in enumerate(solutions)])
            cost = f_values - curr_f_values
            if callable(dist_func):
                cost = np.asarray(dist_func(cost))
            return sum(self.prob * cost)

        model = BaseOptimizer(self._num_scenarios)
        model.set_min_objective(regret)
        model.add_equality_constraint(sum_to_1)
        model.set_bounds(0, 1)

        proportions = self._optimize(model)
        return proportions, proportions @ solutions

    # @property
    # def AP(self):
    #     """
    #     Active Portfolio (AP) objectives.
    #
    #     Active is used when the returns stream of the simulation is the over (under) performance of
    #     the particular asset class over the benchmark. (The first index in the assets axis)
    #
    #     For example, if you have a benchmark (beta) returns stream, 9 other asset classes over
    #     10000 trials and 40 periods, the simulation tensor will be 40 x 10000 x 10 with the first asset
    #     axis being the returns of the benchmark. In such a case, the active portfolio optimizer can
    #     be used to optimize the portfolio relative to the benchmark.
    #     """
    #     return APObjectives(self)
    #
    # @property
    # def PP(self):
    #     """
    #     Policy Portfolio (PP) objectives.
    #
    #     Policy is used on the basic asset classes. For this optimizer, there is an equality constraint set
    #     such that the sum of the weights must be equal to 1. Thus, there is no need to set this equality
    #     constraint.
    #     """
    #     return PPObjectives(self)

    @property
    def max_attempts(self):
        return self._max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int):
        assert isinstance(value, int) and value > 0, 'max_attempts must be an integer >= 1'
        self._max_attempts = value

    def _setup_optimization_model(self, index: int):
        """Sets up the Base Optimizer"""
        model = BaseOptimizer(self._num_assets, self._algorithm, verbose=self._verbose)

        # sets up optimizer's programs and bounds
        for item, set_option in [
            (self._x_tol_abs, model.set_xtol_abs),
            (self._x_tol_rel, model, model.set_xtol_rel),
            (self._f_tol_abs, model.set_ftol_abs),
            (self._f_tol_rel, model.set_ftol_rel),
            (self._max_eval, model.set_maxeval),
            (self._stop_val, model.set_stopval),
            (self._lb, model.set_lower_bounds),
            (self._ub, model.set_upper_bounds)
        ]:
            if item is not None:
                set_option(item)

        # sets constraints
        for constraints, set_constraint in [(self._min.values(), model.add_inequality_matrix_constraint),
                                            (self._meq.values(), model.add_equality_matrix_constraint)]:
            for c in constraints:
                set_constraint(c, self._c_eps)

        for constraints, set_constraint in [(self._heq.values(), model.add_equality_constraint),
                                            (self._hin.values(), model.add_inequality_constraint)]:
            set_constraint(constraints[index], self._c_eps)

        # sets up the objective function
        assert self._max_or_min in ('maximize', 'mimimize') and len(self._obj_fun) == 0, \
            "Objective function is not set yet. Use the .set_max_objective() or .set_min_objective() methods to do so"

        if self._max_or_min == "maximize":
            model.set_max_objective(self._set_gradient(self._obj_fun[index]))
        else:
            model.set_min_objective(self._set_gradient(self._obj_fun[index]))

        return model
