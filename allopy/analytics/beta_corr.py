from typing import NamedTuple

import numpy as np
from copulae.core import cov2corr
from copulae.types import Array

__all__ = ["beta_corr"]


class BetaCorr(NamedTuple):
    beta: float
    corr: float


def beta_corr(cov, w: Array) -> BetaCorr:
    """
    Calculates the beta and correlation of the various assets to the first asset (column)

    Parameters
    ----------
    cov: ndarray
        Covariance matrix of the portfolio

    w: ndarray
        Portfolio weights

    Returns
    -------
    beta: float
        beta to the first asset class
    corr: float
        correlation with the first asset class
    """
    w = np.ravel(w)

    T = np.zeros((2, len(w)))
    T[0, 0] = 1
    T[1] = w

    c = T @ cov @ T.T
    beta = c[1, 0] / c[0, 0]
    corr = cov2corr(c)[1, 0]

    return BetaCorr(beta, corr)
