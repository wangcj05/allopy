from typing import Iterable, Optional, Union

import numpy as np
from copulae.types import Array

from .algorithms import LD_SLSQP
from .base import BaseOptimizer
from .opt_data import OptData

EPSILON = np.finfo('float').eps ** (1 / 3)
OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]


# noinspection PyPep8Naming
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
        if isinstance(data, OptData):
            data = OptData(data, cov_mat, time_unit)

        if cvar_data is None:
            cvar_data = data.copy()
        elif isinstance(cvar_data, OptData):
            cvar_data = OptData(cvar_data, cov_mat, time_unit)

        super().__init__(data.n_assets, algorithm, eps, *args, **kwargs)
        self.data = data
        self.cvar_data = cvar_data

        self._funcs = _Functions(data, cvar_data, rebalance)
        self.AP = _APObjectives(self, self._funcs)

    def adjust_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self._funcs.adjust_returns(eva, vol)
        return self

    def adjust_cvar_returns(self, eva: Optional[Iterable[float]] = None, vol: Optional[Iterable[float]] = None):
        self._funcs.adjust_cvar_returns(eva, vol)
        return self

    @property
    def rebalance(self):
        return self._funcs.rebalance

    @rebalance.setter
    def rebalance(self, rebal: bool):
        assert isinstance(rebal, bool), 'rebalance parameter must be boolean'

        self._funcs.rebalance = rebal


class _APObjectives:
    """
    Active objectives. Active is used when the returns stream of the simulation is the over (under) performance of
    the particular asset class over the benchmark. (The first index in the assets axis)

    For example, if you have a benchmark (beta) returns stream, 9 other asset classes over 10000 trials and 40 periods,
    the simulation tensor will be 40 x 10000 x 10 with the first asset axis being the returns of the benchmark. In
    such a case, the active portfolio optimizer can be used to optimize the portfolio relative to the benchmark.
    """

    def __init__(self, asr: ASROptimizer, funcs: '_Functions'):
        self.asr = asr
        self.funcs = funcs

    def max_eva_st_risk(self, max_te: OptReal = None, max_cvar: OptReal = None, x0: OptArray = None):
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
        opt, f = self.asr, self.funcs

        assert not (max_te is None and max_cvar is None), "If maximizing EVA subject to some sort of TE/CVaR " \
                                                          "constraint, we must at least specify max CVaR or max TE"

        if max_te is not None:
            opt.add_inequality_constraint(f.tracking_error_cons(max_te))

        if max_cvar is not None:
            opt.add_inequality_constraint(f.cvar_cons(max_cvar))

        opt.set_max_objective(f.expected_returns_obj)
        return opt.optimize(x0)

    def min_tracking_error(self, min_ret: OptReal = None, use_active_return=False, x0: OptArray = None):
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
            Otherwise, use the total return. Defaults to True

        x0: ndarray
            Initial vector. Starting position for free variables

        Returns
        -------
        ndarray
            Optimal weights
        """

        opt, f = self.asr, self.funcs

        if min_ret is not None:
            opt.add_inequality_constraint(f.expected_returns_cons(min_ret, use_active_return))

        opt.set_min_objective(f.tracking_error_obj)
        return opt.optimize(x0)

    def min_cvar(self, min_ret: OptReal = None, use_active_return=False, x0: OptArray = None):
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
            Otherwise, use the total return. Defaults to True

        x0: ndarray
            Initial vector. Starting position for free variables

        Returns
        -------
        ndarray
            Optimal weights
        """
        opt, f = self.asr, self.funcs

        if min_ret is not None:
            opt.add_inequality_constraint(f.expected_returns_cons(min_ret, use_active_return))

        opt.set_min_objective(f.cvar_obj)
        return opt.optimize(x0)

    def max_info_ratio(self, x0: OptArray = None):
        """
        Maximizes the information ratio the portfolio.

        :param x0: numeric iterable, optional
            initial vector. Starting position for free variables
        :return: ndarray
            optimal weights
        """

        opt, f = self.asr, self.funcs
        opt.set_max_objective(f.info_ratio_obj)
        return opt.optimize(x0)


def _to_scalar_(name: str):
    """Convenience decorator to convert numbers to floats and raise error if a number is not passed in"""

    def wrapper(func):
        def decorator(cls, value, *args, **kwargs):
            assert np.isscalar(value), f'{name} must be a numeric scalar'

            value = float(value)
            return func(cls, value, *args, **kwargs)

        return decorator

    return wrapper


class _Functions:
    def __init__(self, data: OptData, cvar_data: OptData, rebalance: bool):
        self.data = data
        self.cvar_data = cvar_data
        self._rebal = rebalance

    def adjust_returns(self, eva, vol):
        self.data.calibrate_data(eva, vol, inplace=True)

    def adjust_cvar_returns(self, eva, vol):
        self.cvar_data = self.cvar_data.data.calibrate_data(eva, vol, inplace=True)

    @_to_scalar_('max_cvar')
    def cvar_cons(self, max_cvar: Real):
        """
         Creates a cvar constraint function.

        By default, the constraint function signature is :code:`CVaR(x) >= max_cvar` where `max_cvar` 
        is a negative number. Meaning if you would like to cap cvar at -40%, max_cvar should be set to -0.4.
        
        Parameters
        ----------
        max_cvar: {int, float}
            Maximum cvar.
        """

        def cvar(w):
            return max_cvar - self.data.cvar(w, self.rebalance)

        return cvar

    def cvar_obj(self, w):
        # cvar works in negatives. So to reduce cvar, we have to "maximize" it
        return -self.data.cvar(w, self.rebalance)

    @_to_scalar_('min_ret')
    def expected_returns_cons(self, min_ret: Real, use_active_return: bool):
        def exp_ret(w):
            _w = [0, *w[1:]] if use_active_return else w
            # usually the constraint is >= min_ret, thus need to flip the sign
            return min_ret - self.expected_returns_obj(_w)

        return exp_ret

    def expected_returns_obj(self, w):
        return self.data.expected_return(w, self.rebalance)

    def info_ratio_obj(self, w):
        _w = [0, *w[1:]]
        return self.data.sharpe_ratio(_w, self.rebalance)

    @_to_scalar_('max_te')
    def tracking_error_cons(self, max_te: Real):
        """
        Creates a tracking error function.

        By default, the constraint function signature is :code:`TE(x) <= max_te`. Meaning if you would like to 
        cap tracking error to 3%, max_te should be set to 0.03.
        
        Parameters
        ----------
        max_te: {int, float}
            Maximum tracking error
        """

        def te(w):
            return self.tracking_error_obj(w) - max_te

        return te

    def tracking_error_obj(self, w):
        _w = [0, *w[1:]]
        return self.data.volatility(_w)

    @property
    def rebalance(self):
        return self._rebal

    @rebalance.setter
    def rebalance(self, rebal: bool):
        self._rebal = rebal
