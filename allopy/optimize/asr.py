from typing import Iterable, Optional, Union

import nlopt as nl
import numpy as np
from copulae.types import Array

from .algorithms import LD_SLSQP
from .base import BaseOptimizer
from .obj_ctr import *
from .opt_data import OptData

__all__ = ['ASROptimizer']

OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]


class ASROptimizer(BaseOptimizer):
    def __init__(self,
                 data: Union[np.ndarray, OptData],
                 algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None,
                 rebalance=False,
                 eps: float = np.finfo('float').eps ** (1 / 3),
                 time_unit: int = 'quarterly',
                 *args,
                 **kwargs):
        """
        The ASROptimizer houses several common pre-specified optimization regimes

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

        eps: float
            The tolerance for the optimizer.

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
        if not isinstance(data, OptData):
            data = OptData(data, time_unit)

        if cvar_data is None:
            cvar_data = data.copy().cut_by_horizon(3)
        elif isinstance(cvar_data, (np.ndarray, list, tuple)):
            cvar_data = np.asarray(cvar_data)
            assert cvar_data.ndim == 3, "Must pass in a 3D array for cvar data"
            cvar_data = OptData(cvar_data, time_unit)

        assert isinstance(data, OptData), "data must be an OptData instance"
        assert isinstance(cvar_data, OptData), "cvar_data must be an OptData instance"

        super().__init__(data.n_assets, algorithm, eps, *args, **kwargs)
        self.data = data
        self.cvar_data = cvar_data

        self._rebalance = rebalance

        self._max_attempts = 0
        self.max_attempts = kwargs.get('max_attempts', 100)

    @property
    def AP(self):
        """
        Active objectives.

        Active is used when the returns stream of the simulation is the over (under) performance of
        the particular asset class over the benchmark. (The first index in the assets axis)

        For example, if you have a benchmark (beta) returns stream, 9 other asset classes over
        10000 trials and 40 periods, the simulation tensor will be 40 x 10000 x 10 with the first asset
        axis being the returns of the benchmark. In such a case, the active portfolio optimizer can
        be used to optimize the portfolio relative to the benchmark.
        """
        return APObjectives(self)

    @property
    def PP(self):
        """
        Policy objectives.

        Policy is used on the basic asset classes within GIC. For this optimizer, there is an
        equality constraint set such that the sum of the weights must be equal to 1. Thus, there is
        not need to set the equality constraint yourself.
        """
        return PPObjectives(self)

    def adjust_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self.data = self.data.calibrate_data(eva, vol)
        return self

    def adjust_cvar_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self.cvar_data = self.cvar_data.calibrate_data(eva, vol)
        return self

    @property
    def max_attempts(self):
        return self._max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int):
        assert isinstance(value, int) and value > 0, 'max_attempts must be an integer >= 1'
        self._max_attempts = value

    @property
    def rebalance(self):
        return self._rebalance

    @rebalance.setter
    def rebalance(self, rebal: bool):
        assert isinstance(rebal, bool), 'rebalance parameter must be boolean'
        self._rebalance = rebal


class APObjectives:
    """
    Active objectives. Active is used when the returns stream of the simulation is the over (under) performance of
    the particular asset class over the benchmark. (The first index in the assets axis)

    For example, if you have a benchmark (beta) returns stream, 9 other asset classes over 10000 trials and 40 periods,
    the simulation tensor will be 40 x 10000 x 10 with the first asset axis being the returns of the benchmark. In
    such a case, the active portfolio optimizer can be used to optimize the portfolio relative to the benchmark.

    This is a singleton class meant for easier optimization regime access for the ASROptimizer
    """

    def __init__(self, asr: ASROptimizer):
        self.asr = asr

    def maximize_eva(self, max_te: OptReal = None, max_cvar: OptReal = None, x0: OptArray = None,
                     tol=0.0) -> np.ndarray:
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

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        assert not (max_te is None and max_cvar is None), "If maximizing EVA subject to some sort of TE/CVaR " \
                                                          "constraint, we must at least specify max CVaR or max TE"

        if max_te is not None:
            opt.add_inequality_constraint(tracking_error_ctr(opt.data, max_te), tol)

        if max_cvar is not None:
            opt.add_inequality_constraint(cvar_ctr(opt.cvar_data, max_cvar, opt.rebalance), tol)

        opt.set_max_objective(expected_returns_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)

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
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, use_active_return, opt.rebalance),
                                          tol)

        opt.set_min_objective(tracking_error_obj(opt.data))
        return opt.optimize(x0)

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
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, use_active_return, opt.rebalance),
                                          tol)

        opt.set_min_objective(cvar_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)

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
        opt = self.asr
        opt.set_max_objective(info_ratio_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)


class PPObjectives:
    """
    Policy portfolio objectives.

    Policy portfolio optimization is used only on the asset classes that comprises the policy portfolio. The data
    (simulation) tensor used must be 3 dimensional with data

    For example, if you have a benchmark (beta) returns stream, 9 other asset classes over 10000 trials and 40 periods,
    the simulation tensor will be 40 x 10000 x 10 with the first asset axis being the returns of the benchmark. In
    such a case, the active portfolio optimizer can be used to optimize the portfolio relative to the benchmark.

    This is a singleton class meant for easier optimization regime access for the ASROptimizer
    """

    def __init__(self, asr: ASROptimizer):
        self.asr = asr
        self.asr.add_equality_constraint(lambda w: sum(w) - 1, 1e-6)

    def _optimize(self, x0):
        opt = self.asr

        for _ in range(self.asr.max_attempts):
            try:
                w = opt.optimize(x0)
                if w is not None:
                    return w

            except nl.RoundoffLimited:
                pass
        else:
            if self.asr._verbose:
                print('No solution was found for the given problem. Check the summary() for more information')
            return np.repeat(np.nan, self.asr.data.n_assets)

    def maximize_returns(self, max_vol: OptReal = None, max_cvar: OptReal = None, x0: OptArray = None,
                         tol=0.0) -> np.ndarray:
        """
        Optimizes the expected returns of the portfolio subject to max volatility and/or cvar constraint.
        At least one of the tracking error or cvar constraint must be defined.

        If `max_vol` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
        defined by a positive number. Meaning if you would like to cap tracking error to 3%, max_te should be set to
        0.03.

        Parameters
        ----------
        max_vol: scalar, optional
            Maximum tracking error allowed

        max_cvar: scalar, optional
            Maximum cvar_data allowed

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        assert not (max_vol is None and max_cvar is None), "If maximizing returns subject to some sort of vol/CVaR " \
                                                           "constraint, we must at least specify max CVaR or max vol"

        if max_vol is not None:
            opt.add_inequality_constraint(vol_ctr(opt.data, max_vol), tol)

        if max_cvar is not None:
            opt.add_inequality_constraint(cvar_ctr(opt.cvar_data, max_cvar, opt.rebalance), tol)

        opt.set_max_objective(expected_returns_obj(opt.data, opt.rebalance))
        return self._optimize(x0)

    def minimize_volatility(self, min_ret: OptReal = None, x0: OptArray = None, tol=0.0) -> np.ndarray:
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

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, False, opt.rebalance), tol)

        opt.set_min_objective(vol_obj(opt.data))
        return self._optimize(x0)

    def minimize_cvar(self, min_ret: OptReal = None, x0: OptArray = None, tol=0.0) -> np.ndarray:
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

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, False, opt.rebalance), tol)

        opt.set_min_objective(cvar_obj(opt.data, opt.rebalance))
        return self._optimize(x0)

    def maximize_sharpe_ratio(self, x0: OptArray = None) -> np.ndarray:
        """
        Maximizes the sharpe ratio the portfolio.

        Parameters
        ----------
        x0: array_like, optional
            Initial vector. Starting position for free variables

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr
        opt.set_max_objective(sharpe_ratio_obj(opt.data, opt.rebalance))
        return self._optimize(x0)
