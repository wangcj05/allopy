from typing import Callable, Iterable, List, Optional, Union

import numpy as np

from allopy import OptData, translate_frequency
from allopy.types import OptArray, OptReal
from .constraints import ConstraintBuilder
from .objectives import ObjectiveBuilder
from ..optimizer import RegretOptimizer


class ActivePortfolioRegretOptimizer(RegretOptimizer):
    def __init__(self,
                 data: List[Union[np.ndarray, OptData]],
                 cvar_data: Optional[List[Union[np.ndarray, OptData]]] = None,
                 prob: OptArray = None,
                 rebalance=False,
                 sum_to_1=False,
                 time_unit='quarterly',
                 **kwargs):
        r"""
        PortfolioRegretOptimizer houses several common pre-specified regret optimization routines. Regret optimization
        is a scenario based optimization.

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
        data
            Scenario data. Each data must be a 3 dimensional tensor. Thus data will be a 4-D tensor.

        cvar_data: optional
            CVaR scenario data. Each data must be a 3 dimensional tensor. Thus data will be a 4-D tensor.

        prob
            Vector containing probability of each scenario occurring

        rebalance: bool, optional
            Whether the weights are rebalanced in every time instance. Defaults to False

        sum_to_1:
            If True, portfolio weights must sum to 1. Defaults to False

        time_unit: {int, 'monthly', 'quarterly', 'semi-annually', 'yearly'}, optional
            Specifies how many units (first axis) is required to represent a year. For example, if each time period
            represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
            a month. Alternatively, specify one of 'monthly', 'quarterly', 'semi-annually' or 'yearly'

        kwargs:
            Other keyword arguments to pass to the :class:`RegretOptimizer` base class

        See Also
        --------
        :class:`RegretOptimizer`: RegretOptimizer
        """
        time_unit = translate_frequency(time_unit)
        self._objectives = ObjectiveBuilder(data, cvar_data, rebalance, time_unit)
        self._constraints = ConstraintBuilder(data, cvar_data, rebalance, time_unit)

        na, ns = self._objectives.num_assets, self._objectives.num_scenarios
        super().__init__(na, ns, prob, sum_to_1=sum_to_1, **kwargs)

    @property
    def rebalance(self):
        return self._objectives.rebalance

    @rebalance.setter
    def rebalance(self, value):
        assert isinstance(value, bool), "rebalance must be a boolean value"
        self._objectives.rebalance = value
        self._constraints.rebalance = value

    def maximize_eva(self,
                     max_vol: Optional[Union[float, Iterable[float]]] = None,
                     max_cvar: Optional[Union[float, Iterable[float]]] = None,
                     percentile=5.0,
                     *,
                     as_tracking_error=True,
                     as_active_cvar=False,
                     x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                     x0_prop: OptArray = None,
                     approx=True,
                     dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                     initial_solution: Optional[str] = "random",
                     random_state: Optional[int] = None):
        """
        Optimizes the expected value added of the actively managed portfolio subject to max volatility
        and/or cvar constraint. At least one of the tracking error or cvar constraint must be defined.

        If `max_vol` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
        defined by a positive number. Meaning if you would like to cap tracking error to 3%, max_te should be set to
        0.03.

        Parameters
        ----------
        max_vol: float or list of floats, optional
            Maximum tracking error allowed. If a scalar, the same value will be used for each scenario
            optimization.

        max_cvar: float or list of floats, optional
            Maximum cvar_data allowed. If a scalar, the same value will be used for each scenario optimization.

        percentile: float
            The CVaR percentile value. This means to the expected shortfall will be calculated from values
            below this threshold

        as_active_cvar: bool
            If True, the cvar constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the cvar constraint is calculated using the entire weight vector.

        as_tracking_error: bool
            If True, the volatility constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the volatility constraint is calculated using the entire weight
            vector. This is also known as tracking error.

        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`
        """
        assert not (max_vol is None and max_cvar is None), "If maximizing returns subject to some sort of vol/CVaR " \
                                                           "constraint, we must at least specify max CVaR or max vol"

        if max_vol is not None:
            self.add_inequality_constraint(self._constraints.max_vol(max_vol, as_tracking_error))
        if max_cvar is not None:
            self.add_inequality_constraint(self._constraints.max_cvar(max_cvar, percentile, as_active_cvar))

        self.set_max_objective(self._objectives.max_returns)

        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

    def minimize_volatility(self,
                            min_ret: OptReal = None,
                            *,
                            as_active_return=False,
                            x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                            x0_prop: OptArray = None,
                            approx=True,
                            dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                            initial_solution: Optional[str] = "random",
                            random_state: Optional[int] = None):
        """
        Minimizes the volatility of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float or list of floats, optional
            The minimum returns required for the portfolio. If a scalar, the same value will be used for each
            scenario optimization.

        as_active_return: boolean, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return))

        self.set_min_objective(self._objectives.min_vol(False))
        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

    def minimize_tracking_error(self,
                                min_ret: OptReal = None,
                                *,
                                as_active_return=False,
                                x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                                x0_prop: OptArray = None,
                                approx=True,
                                dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                                initial_solution: Optional[str] = "random",
                                random_state: Optional[int] = None):
        """
        Minimizes the tracking error of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float or list of floats, optional
            The minimum returns required for the portfolio. If a scalar, the same value will be used for each
            scenario optimization.

        as_active_return: boolean, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return))

        self.set_min_objective(self._objectives.min_vol(True))
        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

    def minimize_cvar(self,
                      min_ret: OptReal = None,
                      percentile=5.0,
                      *,
                      as_active_cvar=False,
                      as_active_return=False,
                      x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                      x0_prop: OptArray = None,
                      approx=True,
                      dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                      initial_solution: Optional[str] = "random",
                      random_state: Optional[int] = None) -> np.ndarray:
        """
        Minimizes the conditional value at risk of the portfolio. The present implementation actually minimizes the
        expected shortfall.

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are at least
        as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float or list of floats, optional
            The minimum returns required for the portfolio. If a scalar, the same value will be used for each
            scenario optimization.

        percentile: float
            The CVaR percentile value for the objective. This is the average expected shortfall from values below
            this threshold

        as_active_cvar: bool, optional
            If True, minimizes the active cvar instead of the entire portfolio cvar. If False, minimizes the entire
            portfolio's cvar

        as_active_return: bool, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return))

        self.set_max_objective(self._objectives.max_cvar(percentile, as_active_cvar))
        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

    def maximize_info_ratio(self,
                            *,
                            x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                            x0_prop: OptArray = None,
                            approx=True,
                            dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                            initial_solution: Optional[str] = "random",
                            random_state: Optional[int] = None) -> np.ndarray:
        """
        Maximizes the information ratio the portfolio.

        Parameters
        ----------
        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        self.set_max_objective(self._objectives.max_info_ratio)
        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)

    def maximize_sharpe_ratio(self,
                              *,
                              x0_first_level: Optional[Union[List[OptArray], np.ndarray]] = None,
                              x0_prop: OptArray = None,
                              approx=True,
                              dist_func: Union[Callable[[np.ndarray], np.ndarray], np.ufunc] = np.square,
                              initial_solution: Optional[str] = "random",
                              random_state: Optional[int] = None) -> np.ndarray:
        """
        Maximizes the sharpe ratio the portfolio.

        Parameters
        ----------
        x0_first_level: list of list of floats or ndarray, optional
            List of initial solution vector for each scenario optimization. If provided, the list must have the
            same length at the first dimension as the number of solutions.

        x0_prop: list of floats, optional
            Initial solution vector for the regret optimization (2nd level). This can either be the final
            optimization weights if :code:`approx` is :code:`False` or the scenario proportion otherwise.

        approx: bool
            If True, a linear approximation will be used to calculate the regret optimal

        dist_func: Callable
            A callable function that will be applied as a distance metric for the regret function. The
            default is a quadratic function. See Notes.

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        self.set_max_objective(self._objectives.max_sharpe_ratio)
        return self.optimize(x0_first_level, x0_prop, initial_solution, approx, dist_func, random_state)
