from typing import List, Optional, Union

from copulae.types import Array

from allopy import OptData, get_option

OptArray = Optional[Array]
Real = Union[int, float]  # a real number
OptReal = Optional[Real]

__all__ = ['ctr_max_cvar', 'ctr_max_vol', 'ctr_min_returns', 'obj_max_cvar', 'obj_max_returns', 'obj_max_sharpe_ratio',
           'obj_min_vol', 'sum_equal_1']


def ctr_max_vol(data: List[OptData], max_vol: List[float], active_risk=False):
    """Volatility must be less than max_vol"""

    def make_ctr_max_vol(d: OptData, vol: float):
        def _ctr_max_vol(w):
            w = _active_weights(w, active_risk)
            return get_option("F.SCALE") * (d.volatility(w) - vol)

        return _ctr_max_vol

    return [make_ctr_max_vol(d, v) for d, v in zip(data, max_vol)]


def ctr_max_cvar(data: List[OptData], max_cvar: List[float], rebalance: bool, percentile=5.0, active_cvar=False):
    """CVaR must be greater than max_cvar"""

    def make_ctr_max_cvar(d: OptData, cvar: float):
        def _ctr_max_cvar(w):
            w = _active_weights(w, active_cvar)
            return get_option("F.SCALE") * (cvar - d.cvar(w, rebalance, percentile=percentile))

        return _ctr_max_cvar

    return [make_ctr_max_cvar(d, c) for d, c in zip(data, max_cvar)]


def ctr_min_returns(data: List[OptData], min_ret: List[Real], rebalance=False, active_returns=False):
    """Minimum returns constraint. This is used when objective is to minimize risk st to some minimum returns"""

    def make_ctr_min_returns(d: OptData, r):
        def _ctr_min_returns(w):
            w = _active_weights(w, active_returns)
            return get_option("F.SCALE") * (r - d.expected_return(w, rebalance))

        return _ctr_min_returns

    return [make_ctr_min_returns(d, r) for d, r in zip(data, min_ret)]


def obj_max_cvar(data: List[OptData], rebalance=False, active_cvar=False):
    """Maximizes the CVaR. This means that we're minizming the losses"""

    def make_obj_max_cvar(d: OptData):
        def _obj_max_cvar(w):
            w = _active_weights(w, active_cvar)
            return d.cvar(w, rebalance) * get_option("F.SCALE")

        return _obj_max_cvar

    return [make_obj_max_cvar(d) for d in data]


def obj_max_returns(data: List[OptData], rebalance=False, active_returns=False):
    """Objective function to maximize the returns"""

    def make_obj_max_returns(d: OptData):
        def _obj_max_returns(w):
            w = _active_weights(w, active_returns)
            return d.expected_return(w, rebalance) * get_option("F.SCALE")

        return _obj_max_returns

    return [make_obj_max_returns(d) for d in data]


def obj_max_sharpe_ratio(data: List[OptData], rebalance=False, as_info_ratio=False):
    def make_obj_max_sharpe_ratio(d: OptData):
        def _obj_max_sharpe_ratio(w):
            w = _active_weights(w, as_info_ratio)
            return d.sharpe_ratio(w, rebalance) * get_option("F.SCALE")

        return _obj_max_sharpe_ratio

    return [make_obj_max_sharpe_ratio(d) for d in data]


def obj_min_vol(data: List[OptData], as_tracking_error=False):
    def make_obj_min_vol(d: OptData):
        def _obj_min_vol(w):
            w = _active_weights(w, as_tracking_error)
            return d.volatility(w) * get_option("F.SCALE")

        return _obj_min_vol

    return [make_obj_min_vol(d) for d in data]


def sum_equal_1(w: Array):
    return sum(w) - 1


def _active_weights(weights, active: bool):
    return [0, *weights[1:]] if active else weights
