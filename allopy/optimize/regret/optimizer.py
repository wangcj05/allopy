from typing import Callable, List, Optional, Union

import numpy as np

from allopy import get_option
from allopy.optimize.algorithms import LD_SLSQP
from allopy.optimize.utils import validate_tolerance
from allopy.types import Numeric, OptArray
from ._modelbuilder import ModelBuilder
from ._operations import OptimizationOperation
from .result import RegretOptimizerSolution, RegretResult
from .summary import RegretSummary

__all__ = ['RegretOptimizer']


class RegretOptimizer:
    def __init__(self,
                 num_assets: int,
                 num_scenarios: int,
                 prob: OptArray = None,
                 algorithm=LD_SLSQP,
                 c_eps: Optional[float] = None,
                 xtol_abs: Union[float, np.ndarray, None] = None,
                 xtol_rel: Union[float, np.ndarray, None] = None,
                 ftol_abs: Optional[float] = None,
                 ftol_rel: Optional[float] = None,
                 max_eval: Optional[int] = None,
                 verbose=False,
                 sum_to_1=True,
                 max_attempts=5):
        r"""
        The RegretOptimizer is a convenience class for scenario based optimization.

        Notes
        -----
        The term regret refers to the instance where after having decided on one alternative, the choice
        of a different alternative would have led to a more optimal (better) outcome when the eventual
        scenario transpires.

        The RegretOptimizer employs a 2 stage optimization process. In the first step, the optimizer
        calculates the optimal weights for each scenario. In the second stage, the optimizer minimizes
        the regret function to give the final optimal portfolio weights.

        Assuming the objective is to maximize returns subject to some volatility constraints, the first
        stage optimization will be as listed

        .. math::

            \begin{gather*}
                \underset{w_s}{\max} R_s(w_s)  \forall s \in S \\
                s.t. \\
                \sigma_s(w_s) \leq \Sigma
            \end{gather*}

        where :math:`R_s(\cdot)` is the returns function for scenario :math:`s`, :math:`\sigma_s(\cdot)`
        is the volatility function for scenario :math:`s` and :math:`\Sigma` is the volatility threshold.
        Subsequently, to minimize the regret across all scenarios, :math:`S`,

        .. math::

            \begin{gather*}
                \underset{w}{\min} \sum_{s \in S} p_s \cdot D(R_s(w_s) - R_s(w))
            \end{gather*}

        Where :math:`D(\cdot)` is a distance function (usually quadratic) and :math:`p_s` is the discrete
        probability of scenario :math:`s` occurring.

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

        verbose: bool
            If True, the optimizer will report its operations

        sum_to_1: bool
            If true, the optimal weights for each first level scenario must sum to 1.

        max_attempts: int
            Number of times to retry optimization. This is useful when optimization is in a highly unstable
            or non-convex space.

        See Also
        --------
        :class:`DiscreteUncertaintyOptimizer`: Discrete Uncertainty Optimizer
        """
        assert isinstance(num_assets, int) and num_assets > 0, "num_assets must be an integer and more than 0"
        assert isinstance(num_scenarios, int) and num_scenarios > 0, "num_assets must be an integer and more than 0"
        self._num_assets = num_assets
        self._num_scenarios = num_scenarios

        self._mb = ModelBuilder(
            num_assets,
            num_scenarios,
            algorithm,
            [],
            None,
            None,
            {},
            {},
            {},
            {},
            sum_to_1,
            validate_tolerance(xtol_abs or get_option('EPS.X_ABS')),
            validate_tolerance(xtol_rel or get_option('EPS.X_REL')),
            validate_tolerance(ftol_abs or get_option('EPS.F_ABS')),
            validate_tolerance(ftol_rel or get_option('EPS.F_REL')),
            int(max_eval or get_option('MAX.EVAL')),
            c_eps or get_option('EPS.CONSTRAINT'),
            verbose
        )
        self._prob = None
        self.prob = prob

        # result formatting options
        self._result = None
        self._solution = None

        assert isinstance(max_attempts, int) and max_attempts > 0, 'max_attempts must be an integer >= 1'
        self._max_attempts = max_attempts
        self._verbose = verbose

    @property
    def prob(self):
        """Vector containing probability of each scenario occurring"""
        return self._prob

    @prob.setter
    def prob(self, prob: OptArray):
        if prob is None:
            prob = np.ones(self._num_scenarios) / self._num_scenarios

        assert len(prob) == self._num_scenarios, "probability vector length should equal number of scenarios"
        self._prob = np.asarray(prob)

    @property
    def lower_bounds(self):
        """Lower bound of each variable"""
        return self._mb.lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, lb: Union[int, float, np.ndarray]):
        n = self._num_assets
        if isinstance(lb, (int, float)):
            lb = np.repeat(float(lb), n)

        assert len(lb) == self._mb.num_assets, f"Input vector length must be {n}"
        self._mb.lower_bounds = np.asarray(lb)

    @property
    def upper_bounds(self):
        """Upper bound of each variable"""
        return self._mb.upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, ub: Union[int, float, np.ndarray]):
        n = self._num_assets
        if isinstance(ub, (int, float)):
            ub = np.repeat(float(ub), n)

        assert len(ub) == n, f"Input vector length must be {n}"
        self._mb.upper_bounds = np.asarray(ub)

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
        """
        self.lower_bounds = lb
        self.upper_bounds = ub

        return self

    @property
    def sum_to_1(self):
        return self._mb.sum_to_1

    @sum_to_1.setter
    def sum_to_1(self, value: bool):
        assert isinstance(value, bool), "sum_to_1 must be a boolean value"
        self._mb.sum_to_1 = value

    def set_max_objective(self, functions: List[Callable]):
        """
        Sets the optimizer to maximize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        The list of functions needs to match the number of scenarios. The function at index 0 will be assigned as
        the objective function to the first optimization regime.

        Parameters
        ----------
        functions: List of Callable
            Objective function. The function signature should be such that the first argument takes in a weight
            vector and outputs a numeric (float). The second argument is optional and contains the gradient. If
            given, the user must put adjust the gradients inplace.
        """
        self._mb.max_or_min = 'maximize'
        self._mb.obj_funcs = functions
        return self

    def set_min_objective(self, functions: List[Callable]):
        """
        Sets the optimizer to minimize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        The list of functions needs to match the number of scenarios. The function at index 0 will be assigned as
        the objective function to the first optimization regime.

        Parameters
        ----------
        functions: List of Callable
            Objective function. The function signature should be such that the first argument takes in a weight
            vector and outputs a numeric (float). The second argument is optional and contains the gradient. If
            given, the user must put adjust the gradients inplace.
        """
        self._mb.max_or_min = 'minimize'
        self._mb.obj_funcs = functions
        return self

    def add_inequality_constraint(self, functions: List[Callable]):
        """
        Adds the equality constraint function in standard form, A <= b. If the gradient of the constraint function is
        not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        The list of functions needs to match the number of scenarios. The function at index 0 will be assigned as
        a constraint function to the first optimization regime.

        Parameters
        ----------
        functions: List of Callable
            Constraint functions. The function signature should be such that the first argument takes in a weight
            vector and outputs a numeric (float). The second argument is optional and contains the gradient. If
            given, the user must put adjust the gradients inplace.
        """
        self._mb.add_inequality_constraints(functions)
        return self

    def add_equality_constraint(self, functions: List[Callable]):
        """
        Adds the equality constraint function in standard form, A = b. If the gradient of the constraint function
        is not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        The list of functions needs to match the number of scenarios. The function at index 0 will be assigned as
        a constraint function to the first optimization regime.

        Parameters
        ----------
        functions: List of Callable
            Constraint function. The function signature should be such that the first argument takes in a weight
            vector and outputs a numeric (float). The second argument is optional and contains the gradient. If
            given, the user must put adjust the gradients inplace.
        """
        self._mb.add_equality_constraints(functions)
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
        """
        self._mb.add_inequality_matrix_constraints(A, b)
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
        """
        self._mb.add_equality_matrix_constraints(Aeq, beq)
        return self

    def optimize(self,
                 x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                 x0_prop: OptArray = None,
                 initial_solution: Optional[str] = "random",
                 approx=True,
                 dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                 random_state: Optional[int] = None):
        r"""
        Finds the minimal regret solution across the range of scenarios

        Notes
        -----
        The exact (actual) objective function to minimize regret is given below,

        .. math::

            \begin{gather*}
                \underset{w}{\min} \sum_{s \in S} p_s \cdot D(R_s(w_s) - R_s(w))
            \end{gather*}

        However, given certain problem formulations where the objective and constraint functions are
        linear and convex, the problem can be transformed to

        .. math::

            \begin{gather*}
                \underset{a}{\min} \sum_{s \in S} p_s \cdot D(R_s(w_s) - R_s(W \cdot a))
            \end{gather*}

        where :math:`W` is a matrix where each rows represents a single scenario, :math:`s` and each
        column represents an asset class. This formulation solves for :math:`a` which represents the
        proportion of each scenario that contributes to the final portfolio weights. Thus if there are
        3 scenarios and :math:`a` is :code:`[0.3, 0.5, 0.2]`, it means that the final portfolio took
        30% from scenario 1, 50% from scenario 2 and 20% from scenario 3.

        This formulation makes a strong assumption that the final minimal regret portfolio is a linear
        combination of the weights from each scenario's optimal.

        Notes - Initial Solution
        ------------------------
        The following lists the options for finding an initial solution for the optimization problem. It is best if
        the user supplies an initial value instead of using the heuristics provided if the user already knows the
        region to search.

        random
            Randomly generates "bound-feasible" starting points for the decision variables. Note
            that these variables may not fulfil the other constraints. For problems where the bounds have been
            tightly defined, this often yields a good solution.

        min_constraint_norm
            Solves the optimization problem listed below. The objective is to minimize the :math:`L_2` norm of the
            constraint functions while keeping the decision variables bounded by the original problem's bounds.

            .. math::

                \min | constraint |^2 \\
                s.t. \\
                LB \leq x \leq UB

        Parameters
        ----------
        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied. See notes on
            Initial Solution for more information

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal. See Notes.

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is "random"

        Returns
        -------
        np.ndarray
            Regret optimal solution weights
        """
        opt = OptimizationOperation(self._mb, self.prob, self.max_attempts, self.verbose) \
            .optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

        self._result = opt.result
        self._solution = opt.solution

        return self.solution.regret_optimal

    @property
    def max_attempts(self):
        return self._max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int):
        assert isinstance(value, int) and value > 0, 'max_attempts must be an integer >= 1'
        self._max_attempts = value

    @property
    def result(self) -> RegretResult:
        return self._result

    @property
    def solution(self) -> "RegretOptimizerSolution":
        if self._solution is None:
            raise RuntimeError("Model has not been optimized yet")
        return self._solution

    def set_meta(self, *,
                 asset_names: Optional[List[str]] = None,
                 scenario_names: Optional[List[str]] = None):
        """
        Sets meta data which will be used for result summary

        Parameters
        ----------
        asset_names: list of str, optional
            Names of each asset class

        scenario_names: list of str, optional
            Names of each scenario
        """
        if scenario_names:
            self._result.scenario_names = scenario_names

        if asset_names:
            self._result.asset_names = asset_names

        return self

    @property
    def scenario_names(self):
        return self._result.scenario_names

    @scenario_names.setter
    def scenario_names(self, value: List[str]):
        error = f"scenario_names must be a list with {self._num_scenarios} unique names"
        assert hasattr(value, "__iter__"), error

        value = list(set([str(i) for i in value]))
        assert len(value) == self._num_scenarios, error

        self._result.scenario_names = value

    def summary(self):
        return RegretSummary(self._result)

    def set_epsilon_constraint(self, eps: float):
        """
        Sets the tolerance for the constraint functions

        Parameters
        ----------
        eps: float
            Tolerance
        """
        assert isinstance(eps, float) and eps >= 0, "Epsilon must be a float that is >= 0"
        self._mb.c_eps = eps
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
        """
        self._mb.x_tol_abs = validate_tolerance(tol)
        return self

    def set_xtol_rel(self, tol: Union[float, np.ndarray]):
        r"""
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
        """
        self._mb.x_tol_rel = validate_tolerance(tol)
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
        """
        assert isinstance(n, int), "max evaluation must be an integer"
        self._mb.max_eval = n
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
        """
        self._mb.f_tol_abs = validate_tolerance(tol)
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
       """
        self._mb.f_tol_rel = validate_tolerance(tol)
        return self

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self._verbose = verbose
        assert isinstance(verbose, bool)
