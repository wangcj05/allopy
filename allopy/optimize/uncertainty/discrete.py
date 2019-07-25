import inspect
import re
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import nlopt as nl
import numpy as np
import pandas as pd
from statsmodels.iolib.summary2 import Summary

from allopy import get_option
from allopy.types import Numeric, OptArray
from .result import ConstraintFuncMap, ConstraintMap, Result
from ..algorithms import LD_SLSQP, has_gradient, map_algorithm
from ..utils import *

Cubes = Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]

__all__ = ["Cubes", "DiscreteUncertaintyOptimizer"]


class DiscreteUncertaintyOptimizer(ABC):

    def __init__(self,
                 num_assets: int,
                 num_scenarios: int,
                 prob: OptArray = None,
                 algorithm=LD_SLSQP,
                 auto_grad: Optional[bool] = None,
                 eps_step: Optional[float] = None,
                 c_eps: Optional[float] = None,
                 xtol_abs: Union[float, np.ndarray, None] = None,
                 xtol_rel: Union[float, np.ndarray, None] = None,
                 ftol_abs: Optional[float] = None,
                 ftol_rel: Optional[float] = None,
                 max_eval: Optional[int] = None,
                 stopval: Optional[float] = None,
                 verbose=False):
        r"""
        The DiscreteUncertaintyOptimizer is an abstract optimizer used for optimizing under uncertainty.
        It's main optimization method must be implemented. See the :class:`RegretOptimizer` as an example.

        Parameters
        ----------
        num_assets: int
            Number of assets

        num_scenarios: int
            Number of scenarios

        prob
            Vector containing probability of each scenario occurring

        algorithm
            The optimization algorithm. Default algorithm is Sequential Least Squares Quadratic Programming.

        auto_grad: bool, optional
            If True, the optimizer will calculate numeric gradients for functions that do not specify its
            gradients. The symmetric difference quotient method is used in this case

        eps_step: float, optional
            Epsilon, smallest degree of change, for numeric gradients

        c_eps: float, optional
            Constraint epsilon is the tolerance for the inequality and equality constraints functions. Any
            value that is less than the constraint epsilon is considered to be within the boundary.

        xtol_abs: float or np.ndarray, optional
            Set absolute tolerances on optimization parameters. :code:`tol` is an array giving the
            tolerances: stop when an optimization step (or an estimate of the optimum) changes every
            parameter :code:`x[i]` by less than :code:`tol[i]`. For convenience, if a scalar :code:`tol`
            is given, it will be used to set the absolute tolerances in all n optimization parameters to
            the same value. Criterion is disabled if tol is non-positive.

        xtol_rel: float or np.ndarray, optional
            Set relative tolerance on optimization parameters: stop when an optimization step (or an estimate
            of the optimum) causes a relative change the parameters :code:`x` by less than :code:`tol`,
            i.e. :math:`\|\Delta x\|_w < tol \cdot \|x\|_w` measured by a weighted :math:`L_1` norm
            :math:`\|x\|_w = \sum_i w_i |x_i|`, where the weights :math:`w_i` default to 1. (If there is
            any chance that the optimal :math:`\|x\|` is close to zero, you might want to set an absolute
            tolerance with `code:`xtol_abs` as well.) Criterion is disabled if tol is non-positive.

        ftol_abs: float, optional
            Set absolute tolerance on function value: stop when an optimization step (or an estimate of
            the optimum) changes the function value by less than :code:`tol`. Criterion is disabled if
            tol is non-positive.

        ftol_rel: float, optional
            Set relative tolerance on function value: stop when an optimization step (or an estimate of
            the optimum) changes the objective function value by less than :code:`tol` multiplied by the
            absolute value of the function value. (If there is any chance that your optimum function value
            is close to zero, you might want to set an absolute tolerance with :code:`ftol_abs` as well.)
            Criterion is disabled if tol is non-positive.

        max_eval: int, optional
            Stop when the number of function evaluations exceeds :code:`maxeval`. (This is not a strict
            maximum: the number of function evaluations may exceed :code:`maxeval` slightly, depending
            upon the algorithm.) Criterion is disabled if maxeval is non-positive.

        stopval: int, optional
            Stop when an objective value of at least :code:`stopval` is found: stop minimizing when an
            objective value ≤ :code:`stopval` is found, or stop maximizing a value ≥ :code:`stopval` is
            found. (Setting :code:`stopval` to :code:`-HUGE_VAL` for minimizing or :code:`+HUGE_VAL` for
            maximizing disables this stopping criterion.)

        verbose: bool
            If True, the optimizer will report its operations
        """
        self._algorithm = map_algorithm(algorithm) if isinstance(algorithm, str) else algorithm

        assert isinstance(num_assets, int), "num_assets must be an integer"
        assert isinstance(num_scenarios, int), "num_assets must be an integer"
        self._num_assets = num_assets
        self._num_scenarios = num_scenarios
        self._prob = np.repeat(1 / num_scenarios, num_scenarios)
        self.set_prob(prob)

        has_grad = has_gradient(self._algorithm)
        if has_grad == 'NOT COMPILED':
            raise NotImplementedError(f"Cannot use '{nl.algorithm_name(self._algorithm)}' as it is not compiled")

        # optimizer setup
        self._auto_grad: bool = auto_grad if auto_grad is not None else has_grad
        self._eps: Optional[float] = eps_step or get_option('EPS.STEP')
        self._c_eps: Optional[float] = abs(c_eps or get_option('EPS.CONSTRAINT'))
        self._x_tol_abs: Optional[float] = xtol_abs or get_option('EPS.X_ABS')
        self._x_tol_rel: Optional[float] = xtol_rel
        self._f_tol_abs: Optional[float] = ftol_abs or get_option('EPS.FUNCTION')
        self._f_tol_rel: Optional[float] = ftol_rel
        self._max_eval: Optional[float] = max_eval or get_option('MAX.EVAL')
        self._stop_val: Optional[float] = stopval

        # func
        self._obj_fun: List[Callable[[np.ndarray], float]] = []

        # constraint map
        self._hin: ConstraintFuncMap = {}
        self._heq: ConstraintFuncMap = {}
        self._min: ConstraintMap = {}
        self._meq: ConstraintMap = {}
        self._lb: OptArray = None
        self._ub: OptArray = None

        # result formatting options
        self._result = Result()
        self._max_or_min = None
        self._verbose = verbose

        self._solution = None

    @property
    def prob(self):
        """Vector containing probability of each scenario occurring"""
        return self._prob

    @prob.setter
    def prob(self, prob: OptArray):
        self.set_prob(prob)

    @property
    def lower_bounds(self):
        """Lower bound of each variable"""
        if self._lb is None:
            return np.repeat(-np.inf, self._num_assets)
        return self._lb

    @lower_bounds.setter
    def lower_bounds(self, lb: Union[int, float, np.ndarray]):
        self.set_lower_bounds(lb)

    @property
    def upper_bounds(self):
        """Upper bound of each variable"""
        if self._ub is None:
            return np.repeat(np.inf, self._num_assets)
        return self._ub

    @upper_bounds.setter
    def upper_bounds(self, ub: Union[int, float, np.ndarray]):
        self.set_upper_bounds(ub)

    def set_max_objective(self, fn: Callable, scenarios: Cubes):
        """
        Sets the optimizer to maximize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        Parameters
        ----------
        fn: Callable
            Objective function. The first argument of the function takes in the cube while the second argument
            takes in the weight.

        scenarios
            A list of Monte Carlo simulation cubes, each representing a discrete scenario. This must be a 4
            dimensional object

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._validate_num_scenarios(scenarios)
        self._obj_fun = [self._format_func(fn, cube) for cube in scenarios]

        self._max_or_min = 'maximize'
        self._stop_val = float('inf') if self._stop_val is None else self._stop_val

        return self

    def set_min_objective(self, fn: Callable, scenarios: Cubes):
        """
        Sets the optimizer to minimize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        Parameters
        ----------
        fn: Callable
            Objective function

        scenarios
            A list of Monte Carlo simulation cubes, each representing a discrete scenario. This must be a 4
            dimensional object

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._validate_num_scenarios(scenarios)
        self._obj_fun = [self._format_func(fn, cube) for cube in scenarios]

        self._max_or_min = 'minimize'
        self._stop_val = -float('inf') if self._stop_val is None else self._stop_val

        return self

    def add_inequality_constraint(self, fn: Callable, scenarios: Cubes):
        """
        Adds the equality constraint function in standard form, A <= b. If the gradient of the constraint function is
        not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        Parameters
        ----------
        fn: Callable
            Constraint function

        scenarios
            A list of Monte Carlo simulation cubes, each representing a discrete scenario. This must be a 4
            dimensional object

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._validate_num_scenarios(scenarios)
        self._hin[fn.__name__] = [self._format_func(fn, cube) for cube in scenarios]
        return self

    def add_equality_constraint(self, fn: Callable, scenarios: Cubes):
        """
        Adds the equality constraint function in standard form, A = b. If the gradient of the constraint function
        is not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        Parameters
        ----------
        fn: Callable
            Constraint function

        scenarios
            A list of Monte Carlo simulation cubes, each representing a discrete scenario. This must be a 4
            dimensional object

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._validate_num_scenarios(scenarios)
        self._heq[fn.__name__] = [self._format_func(fn, cube) for cube in scenarios]
        return self

    def add_inequality_matrix_constraint(self, A, b):
        r"""
        Sets inequality constraints in standard matrix form.

        For inequality, :math:`\mathbf{A} \cdot \mathbf{x} \leq \mathbf{b}`

        Parameters
        ----------
        A: {iterable float, ndarray}
            Inequality matrix. Must be 2 dimensional.

        b: {scalar, ndarray}
            Inequality vector or scalar. If scalar, it will be propagated.

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        A, b = validate_matrix_constraints(A, b)

        for i, row, _b in zip(range(len(b)), A, b):
            self._min[f'A_{i}'] = create_matrix_constraint(row, _b)

        return self

    def add_equality_matrix_constraint(self, Aeq, beq):
        r"""
        Sets equality constraints in standard matrix form.

        For equality, :math:`\mathbf{A} \cdot \mathbf{x} = \mathbf{b}`

        Parameters
        ----------
        Aeq: {iterable float, ndarray}
            Equality matrix. Must be 2 dimensional

        beq: {scalar, ndarray}
            Equality vector or scalar. If scalar, it will be propagated

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        Aeq, beq = validate_matrix_constraints(Aeq, beq)

        for i, row, _beq in zip(range(len(beq)), Aeq, beq):
            self._meq[f'Aeq_{i}'] = create_matrix_constraint(row, _beq)

        return self

    def remove_all_constraints(self):
        """Removes all constraints"""
        self._hin = {}
        self._min = {}
        self._heq = {}
        self._meq = {}
        return self

    def set_bounds(self, lb: Numeric, ub: Numeric):
        """
        Sets the lower and upper bound

        Parameters
        ----------
        lb: {int, float, ndarray}
            Vector of lower bounds. If array, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        ub: {int, float, ndarray}
            Vector of upper bounds. If array, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self.set_lower_bounds(lb)
        self.set_upper_bounds(ub)

        return self

    def set_lower_bounds(self, lb: Numeric):
        """
        Sets the lower bounds

        Parameters
        ----------
        lb: {int, float, ndarray}
            Vector of lower bounds. If vector, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        if isinstance(lb, (int, float)):
            lb = np.repeat(float(lb), self._num_assets)

        assert len(lb) == self._num_assets, f"Input vector length must be {self._num_assets}"
        self._lb = np.asarray(lb)
        return self

    def set_upper_bounds(self, ub: Numeric):
        """
        Sets the upper bound

        Parameters
        ----------
        ub: {int, float, ndarray}
            Vector of lower bounds. If vector, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        if isinstance(ub, (int, float)):
            ub = np.repeat(float(ub), self._num_assets)

        assert len(ub) == self._num_assets, f"Input vector length must be {self._num_assets}"
        self._ub = np.asarray(ub)
        return self

    def set_epsilon(self, eps: float):
        """
        Sets the step difference used when calculating the gradient for derivative based optimization algorithms.
        This can ignored if you use a derivative free algorithm or if you specify your gradient specifically.

        Parameters
        ----------
        eps: float
            The gradient step

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        assert isinstance(eps, float) and eps >= 0, "Epsilon must be a float that is >= 0"

        self._eps = eps
        return self

    def set_epsilon_constraint(self, eps: float):
        """
        Sets the tolerance for the constraint functions

        Parameters
        ----------
        eps: float
            Tolerance

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        assert isinstance(eps, float) and eps >= 0, "Epsilon must be a float that is >= 0"

        self._c_eps = eps
        return self

    def set_xtol_abs(self, tol: Union[float, np.ndarray]):
        """
        Sets absolute tolerances on optimization parameters.

        The absolute tolerances on optimization parameters. :code:`tol` is an array giving the
        tolerances: stop when an optimization step (or an estimate of the optimum) changes every
        parameter :code:`x[i]` by less than :code:`tol[i]`. For convenience, if a scalar :code:`tol`
        is given, it will be used to set the absolute tolerances in all n optimization parameters to
        the same value. Criterion is disabled if tol is non-positive.

        Parameters
        ----------
        tol: float or np.ndarray
            Absolute tolerance for each of the free variables

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._x_tol_abs = validate_tolerance(tol)
        return self

    def set_xtol_rel(self, tol: Union[float, np.ndarray]):
        """
        Sets relative tolerances on optimization parameters.

        Set relative tolerance on optimization parameters: stop when an optimization step (or an estimate
        of the optimum) causes a relative change the parameters :code:`x` by less than :code:`tol`,
        i.e. :math:`\|\Delta x\|_w < tol \cdot \|x\|_w` measured by a weighted :math:`L_1` norm
        :math:`\|x\|_w = \sum_i w_i |x_i|`, where the weights :math:`w_i` default to 1. (If there is
        any chance that the optimal :math:`\|x\|` is close to zero, you might want to set an absolute
        tolerance with `code:`xtol_abs` as well.) Criterion is disabled if tol is non-positive.

        Parameters
        ----------
        tol: float or np.ndarray
            relative tolerance for each of the free variables

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._x_tol_rel = validate_tolerance(tol)
        return self

    def set_maxeval(self, n: int):
        """
        Sets maximum number of objective function evaluations.

        Stop when the number of function evaluations exceeds :code:`maxeval`. (This is not a strict
        maximum: the number of function evaluations may exceed :code:`maxeval` slightly, depending
        upon the algorithm.) Criterion is disabled if maxeval is non-positive.

        Parameters
        ----------
        n: int
            maximum number of evaluations

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        assert isinstance(n, int), "max evaluation must be an integer"
        self._max_eval = n
        return self

    def set_ftol_abs(self, tol: float):
        """
        Set absolute tolerance on objective function value.

        The absolute tolerance on function value: stop when an optimization step (or an estimate of
        the optimum) changes the function value by less than :code:`tol`. Criterion is disabled if
        tol is non-positive.

        Parameters
        ----------
        tol: float
            absolute tolerance of objective function value

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._f_tol_abs = validate_tolerance(tol)
        return self

    def set_ftol_rel(self, tol: Optional[float]):
        """
        Set relative tolerance on objective function value.

        Set relative tolerance on function value: stop when an optimization step (or an estimate of
        the optimum) changes the objective function value by less than :code:`tol` multiplied by the
        absolute value of the function value. (If there is any chance that your optimum function value
        is close to zero, you might want to set an absolute tolerance with :code:`ftol_abs` as well.)
        Criterion is disabled if tol is non-positive.

        Parameters
        ----------
        tol: float, optional
            Absolute relative of objective function value

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        self._f_tol_rel = validate_tolerance(tol)
        return self

    def set_stopval(self, stopval: Optional[float]):
        """
        Sets the :code:`stopval`.

        When the objective value of a problem of at least :code:`stopval` is found: stop minimizing
        when an objective value ≤ :code:`stopval` is found, or stop maximizing a value ≥ :code:`stopval`
        is found. (Setting :code:`stopval` to :code:`-HUGE_VAL` for minimizing or :code:`+HUGE_VAL` for
        maximizing disables this stopping criterion.)

        Parameters
        ----------
        stopval: float, optional
            Stopping value

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        assert isinstance(stopval, (int, float)) or stopval is None, "stop value must be a number or None"
        self._stop_val = stopval
        return self

    def set_prob(self, prob: OptArray):
        """
        Sets the probability of each scenario happening. If prob is set to None, it will default to equal
        weighting each scenario

        Parameters
        ----------
        prob
            Vector containing probability of each scenario happening

        Returns
        -------
        DiscreteUncertaintyOptimizer
            Own instance
        """
        if prob is None:
            prob = np.ones(self._num_scenarios) / self._num_scenarios

        self._prob = np.asarray(prob)
        return self

    @property
    def solution(self) -> np.ndarray:
        """
        Returns the solution to the optimization problem

        Returns
        -------
        np.ndarray
            Numpy array of the solution

        Raises
        ------
        RuntimeError
            Model is not optimized yet.
        """
        if self._solution is None:
            raise RuntimeError("Model has not been optimized yet")
        return self._solution

    @abstractmethod
    def optimize(self, x0: OptArray):
        raise NotImplementedError

    def summary(self):
        smry = Summary()
        smry.add_title(re.sub("([A-Z])", r" \1", self.__class__.__name__).strip())

        if self._result.sol is None:
            smry.add_text("Problem has not been optimized yet")
            return smry

        smry.add_df(pd.DataFrame({
            "Assets": [i + 1 for i in range(self._num_assets)],
            "Weight": self._result.sol,
        }))

        if self._result.props:
            smry.add_df(pd.DataFrame({
                "Scenario": [i + 1 for i in range(self._num_scenarios)],
                "Proportion (%)": self._result.props.round(4) * 100
            }))

        smry.add_text("Optimization completed successfully")

        return smry

    @staticmethod
    def _format_func(fn: Callable[[np.ndarray, np.ndarray], float], cube: np.ndarray) -> Callable[[np.ndarray], float]:
        """Formats the objective or constraint function"""
        assert callable(fn), "Argument must be a function"
        f = partial(fn, np.asarray(cube))
        f.__name__ = fn.__name__
        return f

    def _set_gradient(self, fn):
        """Sets a numerical gradient for the function if the gradient is not specified"""
        assert callable(fn), "Argument must be a function"
        if self._auto_grad and len(inspect.signature(fn).parameters) == 1:
            if self._verbose:
                print(f"Setting gradient for function: '{fn.__name__}'")
            return create_gradient_func(fn, self._eps)
        else:
            return fn

    def _validate_num_scenarios(self, scenarios: Cubes):
        error_msg = f"Number of scenarios do not match. Scenarios given: {len(scenarios)}. " \
            f"Scenarios expected: {self._num_scenarios}"
        assert len(scenarios) == self._num_scenarios, error_msg
