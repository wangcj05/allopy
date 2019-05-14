from typing import Iterable, Optional, Union

import numpy as np
from copulae.types import Array

from .algorithms import LD_SLSQP
from .base import BaseOptimizer
from .opt_data import OptData

__all__ = ['ASROptimizer']

EPSILON = np.finfo('float').eps ** (1 / 3)
OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]


class ASROptimizer(BaseOptimizer):
    def __init__(self, data: Union[np.ndarray, OptData], algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None, rebalance=False,
                 eps: float = EPSILON, *args, **kwargs):
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
            copy of the original data. Usually, this is a 3 year :class:`OptData`.

        rebalance: bool, optional
            Whether the weights are rebalanced in every time instance. Defaults to False

        eps: float
            The tolerance for the optimizer.

        args
            other arguments to pass to the :class:`BaseOptimizer`
        kwargs
            other keyword arguments to pass into :class:`OptData` (if you passed in a numpy array for `data`) or into
            the :class:`BaseOptimizer`

        See Also
        --------
        :class:`BaseOptimizer`: Base Optimizer
        :class:`OptData`: Optimizer data wrapper
        """

        cov_mat = kwargs.get('cov_mat', None)
        time_unit = kwargs.get('period_year_length', 4)
        if not isinstance(data, OptData):
            data = OptData(data, cov_mat, time_unit)

        if cvar_data is None:
            cvar_data = data.copy()
        elif not isinstance(cvar_data, OptData):
            cvar_data = OptData(cvar_data, cov_mat, time_unit)

        super().__init__(data.n_assets, algorithm, eps, *args, **kwargs)
        self.data = data
        self.cvar_data = cvar_data

        self._rebalance = rebalance

    @property
    def AP(self):
        """
        Active objectives. Active is used when the returns stream of the simulation is the over (under) performance of
        the particular asset class over the benchmark. (The first index in the assets axis)

        For example, if you have a benchmark (beta) returns stream, 9 other asset classes over 10000 trials and 40 periods,
        the simulation tensor will be 40 x 10000 x 10 with the first asset axis being the returns of the benchmark. In
        such a case, the active portfolio optimizer can be used to optimize the portfolio relative to the benchmark.
        """
        return APObjectives(self)

    @property
    def PP(self):
        return PPObjectives(self)

    def adjust_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self.data = self.data.calibrate_data(eva, vol)
        return self

    def adjust_cvar_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self.cvar_data = self.cvar_data.calibrate_data(eva, vol)
        return self

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

    def maximize_eva(self, max_te: OptReal = None, max_cvar: OptReal = None, x0: OptArray = None) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        assert not (max_te is None and max_cvar is None), "If maximizing EVA subject to some sort of TE/CVaR " \
                                                          "constraint, we must at least specify max CVaR or max TE"

        if max_te is not None:
            opt.add_inequality_constraint(tracking_error_ctr(opt.data, max_te))

        if max_cvar is not None:
            opt.add_inequality_constraint(cvar_ctr(opt.data, max_cvar, opt.rebalance))

        opt.set_max_objective(expected_returns_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)

    def minimize_tracking_error(self, min_ret: OptReal = None, use_active_return=False,
                                x0: OptArray = None) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, use_active_return, opt.rebalance))

        opt.set_min_objective(tracking_error_obj(opt.data))
        return opt.optimize(x0)

    def minimize_cvar(self, min_ret: OptReal = None, use_active_return=False, x0: OptArray = None) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, use_active_return, opt.rebalance))

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

    def maximize_returns(self, max_vol: OptReal = None, max_cvar: OptReal = None, x0: OptArray = None) -> np.ndarray:
        """
        Optimizes the expected returns of the portfolio subject to max volatility and/or cvar constraint.
        At least one of the tracking error or cvar constraint must be defined.

        If `max_te` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        assert not (max_vol is None and max_cvar is None), "If maximizing returns subject to some sort of vol/CVaR " \
                                                           "constraint, we must at least specify max CVaR or max vol"

        if max_vol is not None:
            opt.add_inequality_constraint(vol_ctr(opt.data, max_vol))

        if max_cvar is not None:
            opt.add_inequality_constraint(cvar_ctr(opt.data, max_cvar, opt.rebalance))

        opt.set_max_objective(expected_returns_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)

    def minimize_volatility(self, min_ret: OptReal = None, x0: OptArray = None) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, False, opt.rebalance))

        opt.set_min_objective(vol_obj(opt.data))
        return opt.optimize(x0)

    def minimize_cvar(self, min_ret: OptReal = None, x0: OptArray = None) -> np.ndarray:
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

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt = self.asr

        if min_ret is not None:
            opt.add_inequality_constraint(expected_returns_ctr(opt.data, min_ret, False, opt.rebalance))

        opt.set_min_objective(cvar_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)

    def maximize_sharpe_ratio(self, x0: OptArray) -> np.ndarray:
        """
        Maximizes the sharpe ratio the portfolio.

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
        opt.set_max_objective(sharpe_ratio_obj(opt.data, opt.rebalance))
        return opt.optimize(x0)


def cvar_ctr(data: OptData, max_cvar: Real, rebalance=False):
    """
    Creates a cvar constraint function.

    By default, the constraint function signature is :code:`CVaR(x) >= max_cvar` where `max_cvar`
    is a negative number. Meaning if you would like to cap cvar at -40%, max_cvar should be set to -0.4.
    """

    def cvar(w):
        return float(max_cvar) - data.cvar(w, rebalance)

    return cvar


def cvar_obj(data: OptData, rebalance=False):
    # cvar works in negatives. So to reduce cvar, we have to "maximize" it

    def cvar(w):
        return -data.cvar(w, rebalance)

    return cvar


def expected_returns_ctr(data: OptData, min_ret: Real, use_active_return: bool, rebalance=False):
    exp_ret = expected_returns_obj(data, rebalance)

    def exp_ret_con(w):
        _w = [0, *w[1:]] if use_active_return else w
        # usually the constraint is >= min_ret, thus need to flip the sign
        return float(min_ret) - exp_ret(w)

    return exp_ret_con


def expected_returns_obj(data: OptData, rebalance=False):
    def exp_ret(w):
        return data.expected_return(w, rebalance)

    return exp_ret


def info_ratio_obj(data: OptData, rebalance=False):
    def info_ratio(w):
        _w = [0, *w[1:]]
        return data.sharpe_ratio(_w, rebalance)

    return info_ratio


def sharpe_ratio_obj(data: OptData, rebalance=True):
    def sharpe_ratio(w):
        return data.sharpe_ratio(w, rebalance)

    return sharpe_ratio


def tracking_error_ctr(data: OptData, max_te: Real):
    te_obj = tracking_error_obj(data)

    def te(w):
        return te_obj(w) - float(max_te)

    return te


def tracking_error_obj(data: OptData):
    def te(w):
        return data.volatility([0, *w[1:]])

    return te


def vol_ctr(data: OptData, max_vol: Real):
    v = vol_obj(data)

    def vol(w):
        return v(w) - float(max_vol)

    return vol


def vol_obj(data: OptData):
    def vol(w):
        return data.volatility(w)

    return vol
