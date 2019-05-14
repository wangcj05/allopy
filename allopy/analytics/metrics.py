from typing import Dict, Iterable, Optional

import numpy as np
from copulae.types import Array
from scipy.stats import kurtosis, skew

from allopy.utility import lru_cache_ext
from .utils import calculate_cvar, returns_at, returns_eot, weighted_annualized_returns

Metrics = Dict[str, float]

__all__ = ["performance_ratios_", "performance_ratios", "returns_metrics", "risk_metrics"]


def performance_ratios_(data,
                        w,
                        cov,
                        rebalance,
                        time_unit=4,
                        ow: Optional[Iterable[np.ndarray]] = None,
                        weights: Optional[Iterable[float]] = None,
                        percentile=5.0,
                        cvar_data: Optional[np.ndarray] = None):
    """
    Calculates the performance ratios

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Portfolio weights

    cov: ndarray
        2D matrix which describes the covariance among the asset classes.

    rebalance: boolean
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    time_unit: int, optional
        specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 4 which means 1 period represents
        a quarter

    ow: iterable arrays, optional
        Other weight vectors. Each of these weight vectors must have the same length as :code:`w`.

    weights: iterable float, optional
        the weight with which to weight each weight vectors. The first number refers to :code:`w`'s weight and
        subsequent ones should be the weights of each :code:`ow` vectors.

        For example, if there are 2 weight vectors :code:`w` and :code:`w2` with weights = (0.3, 0.7), then
        :code:`w` will contribute 30% of the weighted returns with :code:`w2` contributing 70%.

    percentile: float, optional
        The percentile to cutoff for CVaR calculation

    cvar_data: np.ndarray, optional
        Tensor of simulated returns which has been adjusted to by CVaR assumptions. The 3 axis
        represents (time, trials, asset) respectively

    Returns
    -------
    dict
        A dictionary containing performance ratios (sharpe, sortino, trasr). Respectively, they are sharpe ratio,
        sortino ratio and tail risk adjusted sharpe ratio

    Examples
    --------
    >>> from allopy.analytics import performance_ratios, returns_metrics, risk_metrics
    >>> from copulae.stats import pearson_rho
    >>> import numpy as np

    >>> np.random.seed(8)
    >>> w = np.random.uniform(0, 1, 5)  # 5 assets
    >>> w /= w.sum()
    >>> data = np.random.normal(size=(20, 100, 5))  # 20 quarters, 100 trials, 5 assets
    >>> cvar_data = data[:12]  # take the first 3 years as cvar data
    >>> cov = np.asarray([pearson_rho(data[i]) for i in range(20)]).mean(0)

    >>> performance_ratios_(data, w, cov, rebalance=False, time_unit=4, cvar_data=cvar_data)
    """
    mean = _calculate_discounted_returns(data, w, rebalance, time_unit, ow, weights)
    std = (w @ cov @ w) ** 0.5
    sstd = _calculate_semi_std(data, w, rebalance, time_unit)

    if cvar_data is None:
        cvar_data = data
    returns = returns_eot(cvar_data, w, rebalance)
    cvar = calculate_cvar(returns, percentile)

    sharpe = mean / std
    sortino = mean / sstd
    trasr = abs(mean / cvar)  # tail risk adjusted sharpe ratio

    return {'sharpe': sharpe, 'sortino': sortino, 'trasr': trasr}


def performance_ratios(returns_metrics_, risk_metrics_):
    """
    Calculates the performance ratios.

    With the returns and risk metrics already calculated, this function will skip additional calculations.

    Parameters
    ----------
    returns_metrics_: dict
        Dictionary containing returns metrics

    risk_metrics_: dict
        Dictionary containing risk metrics

    Returns
    -------
    dict
        A dictionary containing performance ratios (sharpe, sortino, trasr). Respectively, they are sharpe ratio,
        sortino ratio and tail risk adjusted sharpe ratio

    Examples
    --------
    >>> from allopy.analytics import performance_ratios, returns_metrics, risk_metrics
    >>> from copulae.stats import pearson_rho
    >>> import numpy as np

    >>> np.random.seed(8)
    >>> w = np.random.uniform(0, 1, 5)  # 5 assets
    >>> w /= w.sum()
    >>> data = np.random.normal(size=(20, 100, 5))  # 20 quarters, 100 trials, 5 assets
    >>> cvar_data = data[:12]  # take the first 3 years as cvar data
    >>> cov = np.asarray([pearson_rho(data[i]) for i in range(20)]).mean(0)

    >>> returns_metrics_ = returns_metrics(data, w, cov, rebalance=False, time_unit=4)
    >>> risk_metrics_ = risk_metrics(data, w, rebalance=True, percentile=5, time_unit=4, cvar_data=cvar_data)
    >>> performance_ratios(returns_metrics_, risk_metrics_)

    """
    cvar = risk_metrics_['cvar']
    sstd = risk_metrics_['sstd']
    mean = returns_metrics_['mean']
    std = returns_metrics_['std']

    sharpe = mean / std
    sortino = mean / sstd
    trasr = abs(mean / cvar)  # tail risk adjusted sharpe ratio

    return {'sharpe': sharpe, 'sortino': sortino, 'trasr': trasr}


def risk_metrics(data,
                 w,
                 rebalance,
                 percentile=5.0,
                 tail_risk_cutoff=-30.0,
                 time_unit=4,
                 cvar_data=None):
    """
    Calculates the risk metrics

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Asset weight vector

    rebalance: bool
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    percentile: float, optional
        The percentile to cutoff for CVaR calculation

    tail_risk_cutoff: float, optional
        The percentile used to cutoff the tail risk. Defaults at -30% returns

    time_unit: int, optional
        Specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 4 which means 1 period represents
        a quarter

    cvar_data: ndarray, optional
        Tensor of simulated returns which has been adjusted to by CVaR assumptions. The 3 axis
        represents (time, trials, asset) respectively

    Returns
    -------
    dict
        A dictionary containing risk metrics (sstd, cvar, tail_risk)

    Examples
    --------
    >>> from allopy.analytics import risk_metrics
    >>> import numpy as np


    >>> np.random.seed(8)
    >>> w = np.random.uniform(0, 1, 5)  # 5 assets
    >>> w /= w.sum()
    >>> data = np.random.normal(size=(20, 100, 5))  # 20 quarters, 100 trials, 5 assets
    >>> cvar_data = data[:12]  # take the first 3 years as cvar data

    >>> risk_metrics(data, w, rebalance=False, percentile=5, time_unit=4, cvar_data=cvar_data)
    """
    if cvar_data is None:
        cvar_data = data

    returns = returns_eot(cvar_data, w, rebalance)
    cvar = calculate_cvar(returns, percentile)
    tail_risk = _calculate_tail_risk(returns, tail_risk_cutoff)
    sstd = _calculate_semi_std(data, w, rebalance, time_unit)

    return {'cvar': cvar, 'sstd': sstd, 'tail_risk': tail_risk}


def semi_std(data: Array, direction='left', method="full") -> float:
    """
    Calculates the standard deviation. Direction takes in 2 parameters, left calculates the downside deviation
    while up calculates the upside. Method specifies if we should do a full or potential deviation calculation.

    Parameters
    ----------
    data: iterable float
        Vector of floats

    direction: {'left', 'right'}
        Direction to take deviation. 'left' = downside, 'right' = upside

    method: {'full', 'potential'}
        'full' will use the entire vector length to take the mean while 'potential' only uses the filtered vector

    Returns
    -------
    float
        The semi standard deviation for the vector
    """
    data = np.asarray(data)
    direction = direction.lower()
    mean = data.mean()
    if direction == 'left':
        d = data[data < mean]
    elif direction == 'right':
        d = data[data > mean]
    else:
        raise ValueError(f"Unexpected argument '{direction}' for direction. Use either 'left' or 'right'")

    method = method.lower()
    if method == 'full':
        scale = len(data)
    elif method == 'potential':
        scale = len(d)
    else:
        raise ValueError(f"Unexpected argument '{method}' for method. Use either 'full' or 'potential'")

    return (sum((d - mean) ** 2) / scale) ** 0.5


def _calculate_tail_risk(returns, tail_risk_cutoff: float) -> float:
    """Calculates the tail risk of the returns vector at the cutoff point"""
    return sum(returns <= (tail_risk_cutoff / 100)) / len(returns)


def _calculate_semi_std(data: np.ndarray, w: np.ndarray, rebalance: bool, time_unit: float) -> float:
    """Calculates the semi-standard deviation of the data"""
    returns = returns_at(data, w, rebalance)
    return np.apply_along_axis(semi_std, 1, returns).mean() * time_unit ** 0.5


def returns_metrics(data,
                    w,
                    cov,
                    rebalance: bool,
                    time_unit=4,
                    ow: Optional[Iterable[np.ndarray]] = None,
                    weights: Optional[Iterable[float]] = None):
    """
    Calculates the returns metrics

    Parameters
    ----------
    data: ndarray
        Tensor of simulated returns where the 3 axis represents (time, trials, asset) respectively

    w: ndarray
        Portfolio weights

    cov: ndarray
        2D matrix which describes the covariance among the asset classes.

    rebalance: boolean
        If True, the returns is calculated with the portfolio rebalanced at every time instance. Otherwise, the
        portfolio asset returns is allowed to drift

    time_unit: int, optional
        specifies how many units (first axis) is required to represent a year. For example, if each time period
        represents a month, set this to 12. If quarterly, set to 4. Defaults to 4 which means 1 period represents
        a quarter

    ow: array_like, optional
        Other weight vectors. Each of these weight vectors must have the same length as :code:`w`.

    weights: iterable float, optional
        the weight with which to weight each weight vectors. The first number refers to :code:`w`'s weight and
        subsequent ones should be the weights of each :code:`ow` vectors.

        For example, if there are 2 weight vectors :code:`w` and :code:`w2` with weights = (0.3, 0.7), then
        :code:`w` will contribute 30% of the weighted returns with :code:`w2` contributing 70%.

    Returns
    -------
    dict
        A dictionary containing returns metrics (kurtosis, mean, skew, std)

    Examples
    --------
    >>> from allopy.analytics import returns_metrics
    >>> from copulae.stats import pearson_rho
    >>> import numpy as np


    >>> np.random.seed(8)
    >>> w = np.random.uniform(0, 1, 5)  # 5 assets
    >>> w /= w.sum()
    >>> data = np.random.normal(size=(20, 100, 5))  # 20 quarters, 100 trials, 5 assets

    >>> cov = np.asarray([pearson_rho(data[i]) for i in range(20)]).mean(0)
    >>> returns_metrics(data, w, cov, False, time_unit=4)
    {'kurtosis': 53.575811562154286, 'mean': -0.8698538769792689, 'skew': 1.445958384464585, 'std': 0.48492164134823174}
    """
    w = np.ravel(w)

    mean = _calculate_discounted_returns(data, w, rebalance, time_unit, ow, weights)
    std = (w @ cov @ w) ** 0.5
    returns = returns_at(data, w, rebalance)
    _skew = skew(returns, 1).mean()
    _kurt = kurtosis(returns, 1).mean()

    return {'kurtosis': _kurt, 'mean': mean, 'skew': _skew, 'std': std}


@lru_cache_ext()
def _calculate_discounted_returns(data: np.ndarray, w: np.ndarray, rebalance: bool, time_unit: int,
                                  ow: Optional[Iterable[np.ndarray]] = None,
                                  weights: Optional[Iterable[float]] = None):
    """Calculates the discounted returns"""
    if ow is None:
        mean = weighted_annualized_returns(data, rebalance, time_unit, (1, w))
    else:
        assert weights is not None, "weights must be specified if <ow> (other weight) is specified"
        weights = tuple(weights)

        ow = np.asarray(ow)
        if ow.ndim == 1:
            ow = ow.reshape(1, -1)

        assert len(weights) == len(ow) + 1, "weights for <w> (asset weight) and <ow> (other weight) must be specified " \
                                            "if <cw> is specified."
        weight_info = [(_c, _aw) for _c, _aw in zip(weights[1:], ow)]
        mean = weighted_annualized_returns(data, rebalance, time_unit, (weights[0], w), *weight_info)
    return mean
