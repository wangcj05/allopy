import numpy as np
import pandas as pd

from ._base import filepath

__all__ = ["load_index", "load_monte_carlo"]


def load_index(*, download=False) -> pd.DataFrame:
    """
    Dataset contains the index value of 7 asset classes from 01 Jan 1985 to 01 Oct 2017.

    This dataset is usually used only for demonstration purposes. As such, the values have been
    fudged slightly.

    Parameters
    ----------
    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    Returns
    -------
    DataFrame
        A data frame containing the index of the 7 policy asset classes
    """
    return pd.read_csv(filepath('policy_index.csv', download), parse_dates=[0], index_col=0)


def load_monte_carlo(*, download=False, total=False) -> np.ndarray:
    """
    Loads a data set containing a mock Monte Carlo simulation of asset class returns.

    The Monte Carlo tensor has axis represents time, trials and asset respectively. For the
    non-total cube, the shape is 80 x 10000 x 9 meaning there are 80 time periods over
    10000 trials and 9 asset classes.

    The total Monte Carlo tensor's shape is 60 x 10000 x 36

    Parameters
    ----------
    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    total: bool
        If True, loads the monte carlo simulation with the total set of asset classes to simulate a big portfolio

    Returns
    -------
    ndarray
        A Monte Carlo tensor
    """
    filename = 'monte_carlo_total.npy' if total else 'monte_carlo.npy'
    return np.load(filepath(filename, download))
