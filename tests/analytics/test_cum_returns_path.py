import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from allopy.analytics.returns_path import cumulative_returns_path


def test_cum_returns_path(data, w, expected):
    msg = "cumulative returns path does not match"
    n = data.shape[1]

    exp = expected['crp']['no_rebal']
    q = np.asarray(exp.columns * n - 1) / n
    res = cumulative_returns_path(data, w, False, q)

    assert_array_almost_equal(res, exp, 4, f"No rebalance {msg}")

    exp = expected['crp']['rebal']
    q = np.asarray(exp.columns * n - 1) / n
    res = cumulative_returns_path(data, w, True, q)
    assert_array_almost_equal(res, exp, 4, f"Rebalance {msg}")

    # invalid quantile
    with pytest.raises(ValueError):
        cumulative_returns_path(data, w, False, [0, 0.25, 1])

    with pytest.raises(ValueError):
        cumulative_returns_path(data, w, False, [1.1, 0.25])
