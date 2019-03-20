import numpy as np
from copulae.core import cov2corr, near_psd
from copulae.types import Array
from typing import Iterable, NamedTuple, Optional, Union

__all__ = ["beta_corr", "coalesce_covariance_matrix"]


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


def coalesce_covariance_matrix(cov,
                               w: Iterable[float],
                               indices: Optional[Iterable[int]] = None) -> Union[np.ndarray, float]:
    """
    Aggregates the covariance with the weights given at the indices specified

    The aggregated column will be the first column.

    Parameters
    ----------
    cov: ndarray
        Covariance matrix of the portfolio

    w: ndarray
        The weights to aggregate the columns by. Weights do not have to sum to 1, if it needs to, you should check
        it prior

    indices: iterable int, optional
        The column index of the aggregated data. If not specified, method will aggregate the first 'n' columns
        where 'n' is the length of :code:`w`

    Returns
    -------
    ndarray
        Aggregated covariance matrix

    Examples
    --------
    If we have a (60 x 1000 x 10) data and we want to aggregate the assets the first 3 indexes,

    >>> from allopy.analytics.beta_corr import coalesce_covariance_matrix
    >>> import numpy as np

    form covariance matrix
    >>> np.random.seed(8888)
    >>> cov = np.random.standard_normal((5, 5))
    >>> cov = cov @ cov.T

    coalesce first and second column where contribution is (30%, 70%) respectively.
    Does not have to sum to 1
    >>> coalesce_covariance_matrix(cov, (0.3, 0.7))
    coalesce fourth and fifth column
    >>> coalesce_covariance_matrix(cov, (0.2, 0.4), (3, 4))
    """
    w = np.asarray(w)
    cov = np.asarray(cov)
    n = len(w)

    if cov.ndim != 2 or len(cov) != cov.shape[1]:
        raise ValueError('cov must be a square matrix')
    if n > len(cov):
        raise ValueError('adjustment weights cannot be larger than the covariance matrix')

    if indices is None:
        indices = np.arange(n)

    _, a = cov.shape  # get number of assets originally

    # form transform matrix
    T = np.zeros((a - n + 1, a))
    T[0, :n] = w
    T[1:, n:] = np.eye(a - n)

    # re-order covariance matrix
    rev_indices = sorted(set(range(a)) - set(indices))  # these are the indices that are not aggregated
    indices = [*indices, *rev_indices]
    cov = cov[indices][:, indices]  # reorder the covariance matrix, first by rows then by columns

    cov = T @ cov @ T.T
    return float(cov) if cov.size == 1 else near_psd(cov)
