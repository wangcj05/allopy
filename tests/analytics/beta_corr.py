import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from allopy.analytics import beta_corr
from allopy.analytics.utils import coalesce_covariance_matrix


@pytest.fixture()
def cov():
    np.random.seed(8888)
    cov = np.random.standard_normal((5, 5))
    return cov @ cov.T


def test_beta_corr(cov):
    """Tests that beta_corr calculates correctly"""
    w = [0.3, 0.15, 0.2, 0.1, 0.25]
    res = beta_corr(cov, w)
    assert_array_almost_equal(res, (0.2579323022589754, 0.6846726036680288))


def test_coalesce_covariance_matrix(cov):
    """Tests that coalesce_covariance_matrix returns matrix with right shape"""
    new_cov = coalesce_covariance_matrix(cov, (0.3, 0.7))
    assert new_cov.shape == (4, 4)

    new_cov = coalesce_covariance_matrix(cov, (0.3, 0.7), (3, 4))
    assert new_cov.shape == (4, 4)

    new_cov = coalesce_covariance_matrix(cov, (0.3, 0.7, 0.4, 0.5, 0.1))
    assert isinstance(new_cov, float)


def test_coalesce_covariance_matrix_errors():
    """Tests that coalesce_covariance_matrix raises error when arguments are wrong"""
    with pytest.raises(ValueError, match='cov must be a square matrix'):
        coalesce_covariance_matrix(np.random.standard_normal((5, 4)), (0.3, 0.7))

    with pytest.raises(ValueError, match='adjustment weights cannot be larger than the covariance matrix'):
        cov = np.random.standard_normal((5, 5))
        coalesce_covariance_matrix(cov, np.random.uniform(size=6))
