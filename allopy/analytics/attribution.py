import numpy as np

from .utils import returns_eot

__all__ = ["cvar_attr", "returns_attr", "risk_attr"]


def cvar_attr(data, w, rebalance, percentile=5.0) -> np.ndarray:
    """
    Calculates the contribution to CVaR by each asset class
    
    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively
    
    w: ndarray
        Portfolio weights
    
    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift
        
    percentile: float, optional
        The percentile to cutoff for CVaR calculation

    Returns
    -------
    ndarray
        CVaR attribution for each asset class
    """
    assert 0 <= percentile <= 100, "Percentile must be a number between [0, 100]"

    k = int(percentile / 100 * data.shape[1])

    if rebalance:
        t, _, a = data.shape
        order = np.argsort(data @ w, 1)  # this is a matrix
        attr = np.zeros((t, a))
        for i in range(t):
            cvar = data[i, order[i], :][:k].mean(0) * w
            attr[i] = cvar / cvar.sum()

        attr = attr.mean(0)
    else:
        order = np.argsort(returns_eot(data, w, rebalance))  # order for returns
        returns = ((data + 1).prod(0) - 1)[order]  # ordered returns
        cvar = np.mean(returns[:k] * w, 0)
        attr = cvar / cvar.sum()

    return attr


def returns_attr(cov_mat, w, eva=None) -> np.ndarray:
    """
    Calculates the contribution to EVA for each asset class

    Parameters
    ----------
    cov_mat: ndarray
        Portfolio covariance matrix
        
    w: ndarray
        Portfolio weights
        
    eva: ndarray, optional
        array of expected valued added for each asset class

    Returns
    -------
    ndarray
        returns attribution for each asset class
    """
    if eva is None:
        eva = np.zeros(len(w))

    ar = eva + 0.5 * cov_mat.diagonal()
    return (w * ar) / (w @ ar)


def risk_attr(cov_mat, w) -> np.ndarray:
    """
    Calculates the contribution to risk for each asset class
    
    Parameters
    ----------
    cov_mat: ndarray
        Portfolio covariance matrix
        
    w: ndarray
        Portfolio weights

    Returns
    -------
    ndarray
        Risk attribution for each asset class
    """
    vol = w @ cov_mat @ w
    return w * (cov_mat @ w) / vol
