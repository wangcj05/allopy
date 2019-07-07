from typing import Optional, Union

from copulae.types import Array

from allopy import get_option
from .opt_data import OptData

OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]

__all__ = ['ctr_max_cvar', 'ctr_max_vol', 'ctr_min_returns', 'obj_max_cvar', 'obj_max_returns', 'obj_max_sharpe_ratio',
           'obj_min_vol', 'sum_to_1']


def ctr_max_vol(data: OptData, max_vol: float, active_risk=False):
    """Volatility must be less than max_vol"""

    def _ctr_max_vol(w):
        w = _active_weights(w, active_risk)
        return get_option("F.SCALE") * (data.volatility(w) - max_vol)

    return _ctr_max_vol


def ctr_max_cvar(data: OptData, max_cvar: float, rebalance: bool, active_cvar=False):
    """CVaR must be greater than max_cvar"""

    def _ctr_max_cvar(w):
        w = _active_weights(w, active_cvar)
        return get_option("F.SCALE") * (max_cvar - data.cvar(w, rebalance))

    return _ctr_max_cvar


def ctr_min_returns(data: OptData, min_ret: Real, rebalance=False, active_returns=False):
    """Minimim returns constraint. This is used when objective is to minimize risk st to some minimum returns"""

    def _ctr_min_returns(w):
        w = _active_weights(w, active_returns)
        return get_option("F.SCALE") * (min_ret - data.expected_return(w, rebalance))

    return _ctr_min_returns


def obj_max_cvar(data: OptData, rebalance=False, active_cvar=False):
    """Maximizes the CVaR. This means that we're minizming the losses"""

    def _obj_max_cvar(w):
        w = _active_weights(w, active_cvar)
        return data.cvar(w, rebalance) * get_option("F.SCALE")

    return _obj_max_cvar


def obj_max_returns(data: OptData, rebalance=False, active_returns=False):
    """Objective function to maximize the returns"""

    def _obj_max_returns(w):
        w = _active_weights(w, active_returns)
        return data.expected_return(w, rebalance) * get_option("F.SCALE")

    return _obj_max_returns


def obj_max_sharpe_ratio(data: OptData, rebalance=False, as_info_ratio=False):
    def _obj_max_sharpe_ratio(w):
        w = _active_weights(w, as_info_ratio)
        return data.sharpe_ratio(w, rebalance) * get_option("F.SCALE")

    return _obj_max_sharpe_ratio


def obj_min_vol(data: OptData, as_tracking_error=False):
    def _obj_min_vol(w):
        w = _active_weights(w, as_tracking_error)
        return data.volatility(w) * get_option("F.SCALE")

    return _obj_min_vol


def sum_to_1(w: Array):
    return sum(w) - 1


def _active_weights(weights, active: bool):
    return [0, *weights[1:]] if active else weights
