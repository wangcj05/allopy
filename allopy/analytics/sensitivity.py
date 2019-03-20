# TODO maybe check the way we calculate sensitivity
import numpy as np

from allopy.analytics import calculate_cvar, returns_eot


def sensitivity_cvar(cvar_data: np.ndarray, w: np.ndarray, rebalance: bool, percentile=5.0, sens=0.01):
    returns = returns_eot(cvar_data, w, rebalance)
    orig_cvar = calculate_cvar(returns, percentile)

    cvar = np.zeros(len(w))
    for i, ww in enumerate(w):
        w_ = np.asarray([*w])
        w_[i] = ww + sens
        returns = returns_eot(cvar_data, w_, rebalance)
        cvar[i] = calculate_cvar(returns, percentile) - orig_cvar

    return cvar


def sensitivity_returns(eva: np.ndarray, sens=0.01):
    return eva * sens


def sensitivity_vol(cov: np.ndarray, w: np.ndarray, sens=0.01):
    orig_vol = (w @ cov @ w) ** 0.5

    vol = np.zeros(len(w))
    for i, ww in enumerate(w):
        w_ = np.asarray([*w])
        w_[i] = ww + sens

        vol[i] = (w_ @ cov @ w_) ** 0.5 - orig_vol

    # maybe it should be
    # return vol / orig_vol
    return vol
