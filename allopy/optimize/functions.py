import numpy as np

from allopy.analytics.utils import annualize_returns


def cvar(data, w, rebalance, percentile=5.0):
    r"""
    Calculates the CVaR given the weights.

    CVaR is calculated as the mean of returns that is below the percentile specified. Technically, it is the
    expected shortfall. The CVaR is given by:

    .. math::

        ES_\alpha = \frac{\sum_{i \in \mathbf{r}} \mathbb{1}_\alpha (r_i) \cdot r_i}
        {\sum_{i \in \mathbf{r}} \mathbb{1}_\alpha (r_i)}

    where :math:`\alpha` is the percentile cutoff, :math:`\mathbf{r}` is the geometric returns vector and
    :math:`\mathbb{1}_\alpha` is an indicator function specifying if the returns instance, :math:`r_i` is below
    the :math:`alpha` cutoff

    Parameters
    ----------
    data: array_like
        3D simulated returns tensor

    w: array_like
        Portfolio weights

    rebalance: bool
        Whether portfolio is rebalanced every time period

    percentile: float, default 5.0
        The percentile to cutoff for CVaR calculation

    Returns
    -------
    float
        CVaR of the portfolio

    See Also
    --------
    :py:meth:`.portfolio_returns` : Portfolio returns
    """
    assert 0 <= percentile <= 100, "Percentile must be a number between [0, 100]"

    returns = portfolio_returns(data, w, rebalance)
    cutoff = np.percentile(returns, percentile)
    return float(returns[returns <= cutoff].mean())


def expected_return(data, w, rebalance, years):
    r"""
    Calculates the annualized expected return given a weight vector

    The expected annualized returns is given by

    .. math::

        \mu_R = \frac{1}{N} \sum^N_i {r_i^{1/y} - 1}

    where :math:`r` is an instance of the geometric returns vector and :math:`y` is the number of years.

    Parameters
    ----------
    data: array_like
        3D simulated returns tensor

    w: array_like
        Portfolio weights

    rebalance: bool
        Whether portfolio is rebalanced every time period

    years: {float, int}
        Number of years represented by the returns tensor

    Returns
    -------
    float
        Annualized return

    See Also
    --------
    :py:function:`.portfolio_returns` : Portfolio returns
    """
    returns = portfolio_returns(data, w, rebalance) + 1

    return np.mean(annualize_returns(returns, years))


def portfolio_returns(data, w, rebalance):
    r"""
    Calculates the vector of geometric returns of the portfolio for every trial in the simulation.

    The simulated returns is a 3D tensor. If there is rebalancing, then the geometric returns for each trial
    is given by:

    .. math::

        r_i = \prod^T (\mathbf{R_i} \cdot \mathbf{w} + 1)  \forall i \in \{ 1, \dots, N \}

    Otherwise, if there is no rebalancing:

    .. math::

        r_i = (\prod^T (\mathbf{R_i} + 1) - 1) \cdot \mathbf{w}  \forall i \in \{ 1, \dots, N \}

    where :math:`r_i` is the geometric returns for trial :math:`i`, :math:`T` is the total time period,
    :math:`\mathbf{R_i}` is the returns matrix for trial :math:`i`, :math:`\mathbf{w}` is the weights vector
    and :math:`N` is the total number of trials.

    Parameters
    ----------
    data: array_like
        3D simulated returns tensor

    w: array_like
        Portfolio weights

    rebalance: bool
        Whether portfolio is rebalanced every time period

    Returns
    -------
    ndarray
        vector of portfolio returns
    """
    if rebalance:
        return (data @ w + 1).prod(0) - 1
    else:
        return ((data + 1).prod(0) - 1) @ w


def sharpe_ratio(data, w, cov, rebalance, years):
    r"""
    Calculates the portfolio sharpe ratio.

    The formula for the sharpe ratio is given by:

    .. math::

        SR = \frac{\mu_R}{\sigma_R}

    Parameters
    ----------
    data: array_like
        3D simulated returns tensor

    w: array_like
        Portfolio weights

    cov: array_like
        2 dimensional covariance matrix

    rebalance: bool
        Whether portfolio is rebalanced every time period

    years: {float, int}
        Number of years represented by the returns tensor

    Returns
    -------
    float
        Portfolio sharpe ratio

    See Also
    --------
    :py:meth:`.expected_return` : Expected returns
    :py:meth:`.portfolio_returns` : Portfolio returns
    :py:meth:`.volatility` : Volatility

    """
    e = 1e6 * expected_return(data, w, rebalance, years)  # added scale for numerical stability during optimization
    v = 1e6 * volatility(w, cov)

    return e / v


def volatility(w, cov):
    """
    Calculates the volatility of the portfolio given a weight vector. The volatility is given by:

    .. math::

        \mathbf{w} \cdot \Sigma \cdot \mathbf{w^T}

    where :math:`\mathbf{w}` is the weight vector and :math:`\Sigma` is the asset covariance matrix


    Parameters
    ----------
    w: array_like
        1 dimensional portfolio weights

    cov: array_like
        2 dimensional covariance matrix

    Returns
    -------
    float
        Portfolio volatility
    """
    return float(w.T @ cov @ w) ** 0.5
