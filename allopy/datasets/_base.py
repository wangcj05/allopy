import os

import numpy as np
import pandas as pd

__all__ = ["load_index", "load_monte_carlo"]


def _load_file(fn: str):
    return os.path.join(os.path.dirname(__file__), 'data', fn)


def load_index() -> pd.DataFrame:
    """
    Dataset contains the index value of 7 asset classes from 01 Jan 1985 to 01 Oct 2017.

    This dataset is usually used only for demonstration purposes. As such, the values have been
    fudged slightly.

    Returns
    -------
    DataFrame
        A data frame containing the index of the 7 policy asset classes
    """
    fp = _load_file('policy_index.csv')

    return pd.read_csv(fp, parse_dates=[0], index_col=0)


def load_monte_carlo() -> np.ndarray:
    """
    Loads a data set containing a mock Monte Carlo simulation of asset class returns.

    The Monte Carlo tensor has axis represents time, trials and asset respectively. Its shape is
    80 x 10000 x 9 meaning there are 80 time periods over 10000 trials and 9 asset classes.

    Returns
    -------
    ndarray
        A Monte Carlo tensor
    """
    return np.load(_load_file('monte_carlo.npy'))
