from typing import Optional, Union

from copulae.types import Array

from .opt_data import OptData

OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]

__all__ = ['cvar_ctr', 'cvar_obj', 'expected_returns_ctr', 'expected_returns_obj', 'info_ratio_obj', 'sharpe_ratio_obj',
           'tracking_error_obj', 'tracking_error_ctr', 'vol_obj', 'vol_ctr']

CONSTANT = 1e3


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
        return -data.cvar(w, rebalance) * CONSTANT

    return cvar


def expected_returns_ctr(data: OptData, min_ret: Real, use_active_return: bool, rebalance=False):
    exp_ret = expected_returns_obj(data, rebalance)

    def exp_ret_con(w):
        _w = [0, *w[1:]] if use_active_return else w
        # usually the constraint is >= min_ret, thus need to flip the sign
        return float(min_ret) * CONSTANT - exp_ret(w)

    return exp_ret_con


def expected_returns_obj(data: OptData, rebalance=False):
    def exp_ret(w):
        return data.expected_return(w, rebalance) * CONSTANT

    return exp_ret


def info_ratio_obj(data: OptData, rebalance=False):
    def info_ratio(w):
        _w = [0, *w[1:]]
        return data.sharpe_ratio(_w, rebalance) * CONSTANT

    return info_ratio


def sharpe_ratio_obj(data: OptData, rebalance=True):
    def sharpe_ratio(w):
        return data.sharpe_ratio(w, rebalance) * CONSTANT

    return sharpe_ratio


def tracking_error_ctr(data: OptData, max_te: Real):
    te_obj = tracking_error_obj(data)

    def te(w):
        return te_obj(w) - float(max_te) * CONSTANT

    return te


def tracking_error_obj(data: OptData):
    def te(w):
        return data.volatility([0, *w[1:]]) * CONSTANT

    return te


def vol_ctr(data: OptData, max_vol: Real):
    v = vol_obj(data)

    def vol(w):
        return v(w) - float(max_vol) * CONSTANT

    return vol


def vol_obj(data: OptData):
    def vol(w):
        return data.volatility(w) * CONSTANT

    return vol
