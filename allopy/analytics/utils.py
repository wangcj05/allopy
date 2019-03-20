import numpy as np
from copulae.types import Array, Numeric
from typing import Tuple, Union

from allopy.utility.cache import lru_cache_ext

__all__ = ['annualize_returns', 'calculate_cvar', 'returns_at', 'returns_eot', 'weighted_annualized_returns']


def annualize_returns(returns: Numeric, years) -> Union[np.ndarray, float]:
    """
    Annualizes a returns vector

    Parameters
    ----------
    returns: {float, int, ndarray}
        Returns data

    years: float
        Number of years

    Returns
    -------
    {ndarray, float}
        An array of annualized returns
    """
    r = np.asarray(returns) + 1
    return np.sign(r) * np.abs(r) ** (1 / years) - 1


@lru_cache_ext()
def calculate_cvar(returns: np.ndarray, percentile: float) -> float:
    """Calculates the CVaR of the returns vector"""
    cutoff = np.percentile(returns, percentile)
    return returns[returns <= cutoff].mean()


@lru_cache_ext()
def returns_at(data, w, rebalance) -> np.ndarray:
    """
    Calculates the returns across time for each trial

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Portfolio weights

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    Returns
    -------
    ndarray
        A matrix showing the returns across time for each trial. The first axis represents the time and the second axis
        represents the trials
    """
    if rebalance:
        return data @ w
    else:
        # cumulative product along the time axis then dot product with weights
        # this represents how much the portfolio is worth at every time instance across all trials
        value = ((data + 1).cumprod(0) - 1) @ w

        # calculate the returns of the portfolio
        returns = (value[1:] + 1) / (value[:-1] + 1) - 1

        # insert the value at the first time instance
        return np.insert(returns, 0, value[0], axis=0)


@lru_cache_ext()
def returns_eot(data, w, rebalance) -> np.ndarray:
    """
    Calculates the returns at the end of the time (period) for each trial

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Portfolio weights

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    Returns
    -------
    ndarray
        A vector where each element represents the returns for that particular trial of the Monte Carlo simulation at
        the end of the period
    """
    if rebalance:
        return (data @ w + 1).prod(0) - 1
    else:
        return ((data + 1).prod(0) - 1) @ w


def weighted_annualized_returns(data, rebalance, time_unit, *weight_info: Tuple[float, Array]) -> float:
    """
    Calculates the weighted annualized returns.

    This is used when you have different weights and will like to weight returns based on these weights

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    time_unit: int
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4.

    weight_info
        An iterable tuple where the first item is the weight constant and the second item is the asset weights.
        The weight constant must be between 0 and 1 inclusive and sum to 1

    Returns
    -------
    ndarray
        A vector of weighted returns

    Examples
    --------
    >>> from allopy.analytics.utils import weighted_annualized_returns
    >>> import numpy as np

    # generate random weights and data
    >>> data = np.random.standard_normal((20, 1000, 10))  # 5 year quarterly, 1000 trials and 10 assets
    >>> w1, w2 = np.random.uniform(size=10), np.random.uniform(size=10)
    >>> w1 /= sum(w1)
    >>> w2 /= sum(w2)

    # weight returns where w1 contributes 40% and w2 contributes 60%
    >>> weighted_annualized_returns(data, False, 5, (0.4, w1), (0.6, w2))
    """
    if len(weight_info) == 0:
        raise ValueError('weight_info and their constant must be specified')

    weight_info = list(weight_info)  # in case generator is passed in
    wc = np.asarray([w for w, _ in weight_info])  # used to check weight constants are valid

    if np.any(wc < 0) or np.any(wc > 1):
        raise ValueError("weight constant must be between [0, 1]")
    elif not np.isclose(wc.sum(), 1):
        raise ValueError("weight constant must sum to 1")

    ann_ret = 0
    years = len(data) / time_unit
    for c, w in weight_info:
        r = returns_eot(data, w, rebalance)
        ann_ret += c * annualize_returns(r, years).mean()  # weighted sum of annualized returns

    return ann_ret
