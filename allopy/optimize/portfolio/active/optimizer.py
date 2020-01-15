from typing import Optional, Union

import numpy as np

from allopy import OptData
from allopy.optimize.algorithms import LD_SLSQP
from allopy.types import OptArray, OptReal
from .constraints import ConstraintBuilder
from .objectives import ObjectiveBuilder
from ..abstract import AbstractPortfolioOptimizer


class ActivePortfolioOptimizer(AbstractPortfolioOptimizer):
    def __init__(self,
                 data: Union[np.ndarray, OptData],
                 algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None,
                 rebalance=False,
                 time_unit='quarterly',
                 sum_to_1=False,
                 *args,
                 **kwargs):
        """
        The ActivePortfolioOptimizer houses several common pre-specified optimization routines.

        ActivePortfolioOptimizer assumes that the optimization model has no uncertainty. That is, the
        portfolio is expected to undergo a single fixed scenario in the future.

        Notes
        -----
        ActivePortfolioOptimizer is a special case of the PortfolioOptimizer where the goal is to determine
        the best mix of of the portfolio relative to ba benchmark. By convention, the first asset of
        the data is the benchmark returns stream. The remaining returns stream is then the over or under
        performance of the returns over the benchmark. In this way, the optimization has an intuitive meaning
        of allocating resources whilst taking account

        For example, if you have a benchmark (beta) returns stream, 9 other asset classes over
        10000 trials and 40 periods, the simulation tensor will be 40 x 10000 x 10 with the first asset
        axis being the returns of the benchmark. In such a case, the active portfolio optimizer can
        be used to optimize the portfolio relative to the benchmark.

        Parameters
        ----------
        data: {ndarray, OptData}
            The data used for optimization

        algorithm: {int, string}
            The algorithm used for optimization. Default is Sequential Least Squares Programming

        cvar_data: {ndarray, OptData}
            The cvar_data data used as constraint during the optimization. If this is not set, will default to being a
            copy of the original data that is trimmed to the first 3 years. If an array like object is passed in,
            the data must be a 3D array with axis representing time, trials and assets respectively. In that
            instance, the horizon will not be cut at 3 years, rather it'll be left to the user.

        rebalance: bool, optional
            Whether the weights are rebalanced in every time instance. Defaults to False

        time_unit: {int, 'monthly', 'quarterly', 'semi-annually', 'yearly'}, optional
            Specifies how many units (first axis) is required to represent a year. For example, if each time period
            represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
            a month. Alternatively, specify one of 'monthly', 'quarterly', 'semi-annually' or 'yearly'

        sum_to_1: bool
            If False, the weights do not need to sum to 1. This should be False for active optimizer.

        args:
            other arguments to pass to the :class:`BaseOptimizer`

        kwargs:
            other keyword arguments to pass into :class:`OptData` (if you passed in a numpy array for `data`) or into
            the :class:`BaseOptimizer`

        See Also
        --------
        :class:`BaseOptimizer`: Base Optimizer
        :class:`OptData`: Optimizer data wrapper
        """
        super().__init__(data, algorithm, cvar_data, rebalance, time_unit, sum_to_1=sum_to_1, *args, **kwargs)
        self._objectives = ObjectiveBuilder(self.data, self.cvar_data, rebalance)
        self._constraints = ConstraintBuilder(self.data, self.cvar_data, rebalance)

    def maximize_eva(self, max_vol: OptReal = None, max_cvar: OptReal = None, percentile=5.0, x0: OptArray = None, *,
                     as_tracking_error=True, as_active_cvar=False, tol=0.0) -> np.ndarray:
        """
        Optimizes the expected value added of the portfolio subject to max tracking error and/or cvar constraint.
        At least one of the tracking error or cvar constraint must be defined.

        If `max_te` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
        defined by a positive number. Meaning if you would like to cap tracking error to 3%, max_te should be set to
        0.03.

        Parameters
        ----------
        max_vol: float, optional
            Maximum tracking error allowed

        max_cvar: float, optional
            Maximum cvar_data allowed

        percentile: float
            The CVaR percentile value. This means to the expected shortfall will be calculated from values
            below this threshold

        x0: ndarray
            Initial vector. Starting position for free variables

        as_active_cvar: bool
            If True, the cvar constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the cvar constraint is calculated using the entire weight vector.

        as_tracking_error: bool
            If True, the volatility constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the volatility constraint is calculated using the entire weight
            vector. This is also known as tracking error.

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        assert not (max_vol is None and max_cvar is None), \
            "If maximizing EVA subject to some sort of TE/CVaR constraint, we must at least specify max CVaR or max TE"

        if max_vol is not None:
            self.add_inequality_constraint(self._constraints.max_vol(max_vol, as_tracking_error), tol)

        if max_cvar is not None:
            self.add_inequality_constraint(self._constraints.max_cvar(max_cvar, percentile, as_active_cvar))

        self.set_max_objective(self._objectives.max_returns)
        return self.optimize(x0)

    def minimize_tracking_error(self, min_ret: OptReal = None, x0: OptArray = None, *, as_active_return=False,
                                tol=0.0) -> np.ndarray:
        """
        Minimizes the tracking error of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        x0: ndarray
            Initial vector. Starting position for free variables

        as_active_return: boolean, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return), tol)

        self.set_min_objective(self._objectives.min_volatility(is_tracking_error=True))
        return self.optimize(x0)

    def minimize_volatility(self, min_ret: OptReal = None, x0: OptArray = None, *, as_active_return=False,
                            tol=0.0) -> np.ndarray:
        """
        Minimizes the volatility of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        x0: ndarray
            Initial vector. Starting position for free variables


        as_active_return: boolean, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return), tol)

        self.set_min_objective(self._objectives.min_volatility(is_tracking_error=False))
        return self.optimize(x0)

    def minimize_cvar(self, min_ret: OptReal = None, x0: OptArray = None, *, percentile=5.0, as_active_cvar=False,
                      as_active_return=False, tol=0.0) -> np.ndarray:
        """
        Minimizes the conditional value at risk of the portfolio. The present implementation actually minimizes the
        expected shortfall.

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are at least
        as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        x0: ndarray
            Initial vector. Starting position for free variables

        percentile: float
            The CVaR percentile value for the objective. This means to the expected shortfall will be calculated
            from values below this threshold

        as_active_cvar: bool, optional
            If True, minimizes the active cvar instead of the entire portfolio cvar. If False, minimizes the entire
            portfolio's cvar

        as_active_return: bool, optional
            If True, the returns constraint is calculated using the active portion of the weights. That is, the
            first value is forced to 0. If False, the returns constraint is calculated using the entire weight
            vector.

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(self._constraints.min_returns(min_ret, as_active_return), tol)

        self.set_max_objective(self._objectives.max_cvar(as_active_cvar, percentile))
        return self.optimize(x0)

    def maximize_info_ratio(self, x0: OptArray = None) -> np.ndarray:
        """
        Maximizes the information ratio the portfolio.

        Parameters
        ----------
        x0: array_like, optional
            initial vector. Starting position for free variables

        Returns
        -------
        ndarray
            Optimal weights
        """
        self.set_max_objective(self._objectives.max_info_ratio)
        return self.optimize(x0)

    def maximize_sharpe_ratio(self, x0: OptArray = None) -> np.ndarray:
        """
        Maximizes the Sharpe ratio the portfolio.

        Parameters
        ----------
        x0: array_like, optional
            initial vector. Starting position for free variables

        Returns
        -------
        ndarray
            Optimal weights
        """
        self.set_max_objective(self._objectives.max_sharpe_ratio)
        return self.optimize(x0)
