from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from copulae.types import Array

__all__ = ["drawdowns", "drawdown_stats"]

DateEquivalent = Optional[Union[Iterable[str], Iterable[int]]]


def drawdowns(data: Union[pd.Series, pd.DataFrame, Array],
              dates: DateEquivalent = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute series of drawdowns from financial returns

    Parameters
    ----------
    data: {Series, DataFrame, Iterable float}
        A vector or matrix of returns

    dates
        An iterable of dates in string form. If :code:`data` argument is pandas Series or DataFrame,
        this parameter may be omitted as it can be derived from the index.

    Returns
    -------
    {Series, DataFrame}
        If input is a vector, returns a Series. Otherwise a DataFrame. The index will be the dates.
    """
    if isinstance(data, pd.Series):
        if dates is None:
            dates = list(data.index)
        dd = pd.Series(_univariate_drawdown(data))
    elif isinstance(data, pd.DataFrame):
        if dates is None:
            dates = list(data.index)
        dd = pd.DataFrame({col: _univariate_drawdown(data[col]) for col in data.columns})
    else:
        data = np.asarray(data)
        if dates is None:
            dates = np.arange(len(data))

        if data.ndim == 1:
            dd = pd.Series(_univariate_drawdown(data))
        elif data.ndim == 2:
            dd = pd.DataFrame({f'Asset_{i}': _univariate_drawdown(data[:, i]) for i in range(data.shape[1])})
        else:
            raise ValueError('data can only be 1 or 2 dimensional')

    dd.index = dates
    # mark an attribute inside the series or data frame so that drawdown_stats can differentiate
    # between a normal series or data frame and one that has gone through the drawdowns functions
    dd.is_drawdown = True
    return dd


def _univariate_drawdown(x: Array):
    """Calculates the drawdown of a vector"""
    cum_returns = np.cumprod(1 + x)
    cum_max = np.maximum.accumulate(cum_returns)
    return cum_returns / cum_max - 1


def drawdown_stats(x: Union[pd.Series, Array], dates: DateEquivalent = None) -> pd.DataFrame:
    """
    Computes the drawdown statistics

    Parameters
    ----------
    x: {Series, iterable float}
        A vector of drawdown or returns. If returns is passed in, drawdown will be calculated automatically

    dates
        An iterable of dates in string form. If :code:`data` argument is pandas Series or DataFrame,
        this parameter may be omitted as it can be derived from the index.

    Returns
    -------
    DataFrame
        A data frame containing drawdown statistics
    """
    if dates is None:
        if type(x) in (pd.DataFrame, pd.Series):
            dates = x.index
        else:
            dates = np.arange(len(x))

    dates = np.asarray(dates)

    if not hasattr(x, 'is_drawdown'):
        x = drawdowns(x, dates)

    if x.ndim != 1:
        raise ValueError('drawdown data <x> can only be 1 dimensional')

    from_, trough_, to_, drawdown_ = [], [], [], []

    # initial conditions
    prior_sign = 1 if x[0] >= 0 else 0
    dd_start = 0  # drawdown start index
    dd_min = 0  # drawdown trough index
    so_far = x[0]  # returns so far

    for i in range(1, len(x)):
        curr_ret = x[i]
        curr_sign = 0 if curr_ret < 0 else 1

        if curr_sign == prior_sign:
            # still falling, continue to capture biggest drawdown
            if curr_ret < so_far:
                so_far = curr_ret
                dd_min = i
        else:
            # recovery occurred
            drawdown_.append(so_far)  # note the biggest drawdown
            from_.append(dd_start)  # take it's starting position
            trough_.append(dd_min)  # take biggest drawdown's index
            to_.append(i)  # take drawdown's stop index

            dd_start = i  # reset drawdown start index, this is the new 'high'
            so_far = curr_ret
            dd_min = i  # set new biggest drawdown index
            prior_sign = curr_sign

    drawdown_.append(so_far)
    from_.append(dd_start)
    trough_.append(dd_min)
    to_.append(len(x) - 1)

    drawdown_ = np.array([*drawdown_, so_far])
    from_ = np.array([*from_, dd_start])
    trough_ = np.array([*trough_, dd_min])
    to_ = np.array([*to_, len(x) - 1])

    if to_[-1] > len(dates):
        to_t = np.array([*to_, to_[-1] - 1])
        to_r = np.array([*to_, np.nan])
    else:
        to_t, to_r = to_, to_

    stats = pd.DataFrame({
        'from': dates[from_],
        'trough': dates[trough_],
        'to': dates[to_t],
        'drawdown': drawdown_,
        'length': to_ - from_ + 1,
        'peak_to_trough': trough_ - from_ + 1,
        'recovery': to_r - trough_
    })

    return stats[stats['drawdown'] < 0].sort_values(by=['drawdown']).reset_index(drop=True)
