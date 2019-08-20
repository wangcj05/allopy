from typing import Optional, Union

import numpy as np

from allopy import OptData
from allopy.types import OptArray, OptReal
from .abstract import AbstractPortfolioOptimizer
from .obj_ctr import *
from ..algorithms import LD_SLSQP


class ActivePortfolioOptimizer(AbstractPortfolioOptimizer):
    def __init__(self,
                 data: Union[np.ndarray, OptData],
                 algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None,
                 rebalance=False,
                 time_unit='quarterly',
                 *args,
                 **kwargs):
        """
        The ActivePortfolioOptimizer houses several common pre-specified optimization routines.

        ActivePortfolioOptimizer assumes that the optimization model has no uncertainty. That is, the
        portfolio is expected to undergo a single fixed scenario in the future. By default, the
        ActivePortfolioOptimizer will automatically add an equality constraint that forces the portfolio
        weights to sum to 1.

        Notes
        -----
        ActivePortfolioOptimizer is a special case of the PortfolioOptimizer where the goal is to determine
        the best mix of of the portfolio relative to ba benchmark. By convention, the first asset should of
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
        super().__init__(data, algorithm, cvar_data, rebalance, time_unit, False, *args, **kwargs)

    def maximize_eva(self, max_te: OptReal = None, max_cvar: OptReal = None, percentile=5.0, x0: OptArray = None,
                     active_cvar=False, tol=0.0) -> np.ndarray:
        """
        Optimizes the expected value added of the portfolio subject to max tracking error and/or cvar constraint.
        At least one of the tracking error or cvar constraint must be defined.

        If `max_te` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
        defined by a positive number. Meaning if you would like to cap tracking error to 3%, max_te should be set to
        0.03.

        Parameters
        ----------
        max_te: float, optional
            Maximum tracking error allowed

        max_cvar: float, optional
            Maximum cvar_data allowed

        percentile: float
            The CVaR percentile value. This means to the expected shortfall will be calculated from values
            below this threshold

        x0: ndarray
            Initial vector. Starting position for free variables

        active_cvar: bool
            If True, calculates the CVaR of only the active portfolio of the portfolio. If False, calculates the
            CVaR of the entire portfolio.

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        assert not (max_te is None and max_cvar is None), "If maximizing EVA subject to some sort of TE/CVaR " \
                                                          "constraint, we must at least specify max CVaR or max TE"

        if max_te is not None:
            self.add_inequality_constraint(ctr_max_vol(self.data, max_te, True), tol)

        if max_cvar is not None:
            self.add_inequality_constraint(ctr_max_cvar(self.cvar_data,
                                                        max_cvar,
                                                        self.rebalance,
                                                        percentile,
                                                        active_cvar))

        self.set_max_objective(obj_max_returns(self.data, self.rebalance, True))
        return self.optimize(x0)

    def minimize_tracking_error(self, min_ret: OptReal = None, use_active_return=False, x0: OptArray = None,
                                tol=0.0) -> np.ndarray:
        """
        Minimizes the tracking error of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        use_active_return: boolean, optional
            If True, return is calculated as active return, that is the first (passive) weight will be set to 0.
            Otherwise, use the total return. Defaults to True. This is important in that the min_ret parameter
            should reflect pure alpha as all the beta in the passive have been stripped away when this argument
            is True.

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(ctr_min_returns(self.data, min_ret, self.rebalance, use_active_return), tol)

        self.set_min_objective(obj_min_vol(self.data, as_tracking_error=True))
        return self.optimize(x0)

    def minimize_cvar(self, min_ret: OptReal = None, use_active_return=False, x0: OptArray = None,
                      tol=0.0) -> np.ndarray:
        """
        Minimizes the conditional value at risk of the portfolio. The present implementation actually minimizes the
        expected shortfall.

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are at least
        as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        use_active_return: boolean, optional
            If True, return is calculated as active return, that is the first (passive) weight will be set to 0.
            Otherwise, use the total return. Defaults to True. This is important in that the min_ret parameter
            should reflect pure alpha as all the beta in the passive have been stripped away when this argument
            is True.

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(ctr_min_returns(self.data, min_ret, self.rebalance, use_active_return), tol)

        self.set_max_objective(obj_max_cvar(self.data, self.rebalance))
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
        self.set_max_objective(obj_max_sharpe_ratio(self.data, self.rebalance, as_info_ratio=True))
        return self.optimize(x0)
