# TODO maybe check the way we calculate sensitivity
import numpy as np

from .utils import calculate_cvar, returns_eot


def sensitivity_cvar(cvar_data: np.ndarray, w: np.ndarray, rebalance: bool, percentile=5.0, sens=0.01):
    """
    Calculates the sensitivity of the change in CVaR with a unit (`sens`) change in the asset weight for each asset.

    Parameters
    ----------
    cvar_data: ndarray
        3D Tensor containing asset returns with the EVA calibrated

    w: ndarray
        Portfolio weights

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    percentile: float, optional
        The percentile to cutoff for CVaR calculation

    sens: float
        Sensitivity of change. Defaults to 0.01, which represents a 1% absolute change in weight

    Returns
    -------
    ndarray
        CVaR sensitivity of each asset class
    """
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
    """
    Calculates the sensitivity of the change in the returns with a unit (`sens`) change in the asset weight for each
    asset.

    Parameters
    ----------
    eva: ndarray
        Expected value added

    sens: float
        Sensitivity of change. Defaults to 0.01, which represents a 1% absolute change in weight

    Returns
    -------
    ndarray
        Returns sensitivity of each asset class
    """
    return eva * sens


def sensitivity_vol(cov: np.ndarray, w: np.ndarray, sens=0.01):
    """
    Calculates the sensitivity of the change in the volatility with a unit (`sens`) change in the asset weight for each
    asset.

    Parameters
    ----------
    cov: ndarray
        2D matrix which describes the covariance among the asset classes.

    w: ndarray
        Portfolio weights

    sens: float
        Sensitivity of change. Defaults to 0.01, which represents a 1% absolute change in weight

    Returns
    -------
    ndarray
        Volatility sensitivity of each asset class
    """
    orig_vol = (w @ cov @ w) ** 0.5

    vol = np.zeros(len(w))
    for i, ww in enumerate(w):
        w_ = np.asarray([*w])
        w_[i] = ww + sens

        vol[i] = (w_ @ cov @ w_) ** 0.5 - orig_vol

    # maybe it should be
    # return vol / orig_vol
    return vol
