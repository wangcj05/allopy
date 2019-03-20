import numpy as np
from copulae.types import Array
from typing import Optional

from .utils import returns_eot

__all__ = ["cumulative_returns_path"]


def cumulative_returns_path(data, w, rebalance: bool, quantile: Optional[Array] = None):
    """
    Calculates the cumulative returns path.

    Essentially, the CRP is visualization of the different ways the portfolio could go based on the Monte Carlo
    simulation.

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Portfolio weights

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    quantile: iterable float
        An 1D array indicating the CRT quantile of interest. If not specified, a quantile is generated

    Returns
    -------
    ndarray
        A 2D matrix where the first axis represents the time and the second axis represents the portfolio returns at
        that time period
    """

    # calculates the returns and gets the order of the returns
    order = np.argsort(returns_eot(data, w, rebalance))

    # form the quantile
    n = len(order)
    if quantile is None:
        # default quantile formulation
        base = np.array([*(0.05, 0.25, 0.50, 0.75, 0.95), *np.linspace(0.25, 0.975, num=100)]) - 1  # the base quantile
        quantile = np.sort(np.unique(np.asarray(base * n, int)))
    else:
        # validating and prepping quantile
        quantile = np.asarray(quantile)
        if not np.alltrue((quantile <= 1) & (quantile > 0)):
            raise ValueError('quantile should be a vector which contains value within the range (0, 1]')

        quantile = np.asarray(quantile * n, int)

    # reorder returns based on order calculated previously and then extracting the quantile
    returns = data[:, order[quantile], :]

    if rebalance:
        return (returns @ w + 1).cumprod(0) - 1
    else:
        return ((returns + 1).cumprod(0) - 1) @ w
