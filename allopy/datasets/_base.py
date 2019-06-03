import os

import pandas as pd

__all__ = ["load_policy_index"]


def _load_file(fn: str):
    return os.path.join(os.path.dirname(__file__), 'data', fn)


def load_policy_index() -> pd.DataFrame:
    """
    Dataset contains the index value of the 7 policy asset classes from 01 Jan 1985 to 01 Oct 2017.

    This dataset is usually used only for demonstration purposes. As such, the values have been
    fudged slightly.

    Returns
    -------
    DataFrame
        A data frame containing the index of the 7 policy asset classes
    """
    fp = _load_file('policy_index.csv')

    return pd.read_csv(fp, parse_dates=[0], index_col=0)
