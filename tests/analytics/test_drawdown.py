import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pandas.api.types import is_numeric_dtype

from allopy.analytics.drawdown import drawdown_stats, drawdowns
from tests.analytics.setup import read_historical


@pytest.fixture()
def hist_pp():
    hist_pp = read_historical('passive')
    return hist_pp


@pytest.fixture()
def expected_pp():
    return read_historical('expected', 'pp-drawdown')


def test_drawdown_1d(hist_pp, expected_pp, w):
    exp = expected_pp.Portfolio

    dates = hist_pp.DATE
    x = hist_pp.iloc[:, 1:] @ w
    res = drawdowns(x, dates)

    assert np.all(res.index == dates)
    assert_array_almost_equal(res, exp)

    x.index = dates
    res = drawdowns(x)
    assert_array_almost_equal(res, exp)

    arr_x = np.asarray(x)
    assert_array_almost_equal(drawdowns(arr_x), exp)
    assert_array_almost_equal(drawdowns(arr_x, dates), exp)


def test_drawdown_2d(hist_pp, expected_pp):
    exp = expected_pp.iloc[:, 1:-1]

    dates = hist_pp.DATE
    x = hist_pp.iloc[:, 1:]
    res = drawdowns(x, dates)

    assert np.all(res.index == dates)
    assert_array_almost_equal(res, exp)

    x.index = dates
    res = drawdowns(x)
    assert_array_almost_equal(res, exp)

    arr_x = np.asarray(x)
    assert_array_almost_equal(drawdowns(arr_x), exp)
    assert_array_almost_equal(drawdowns(arr_x, dates), exp)


def test_drawdown_stats(hist_pp, w):
    exp = read_historical('expected', 'pp-drawdown-stats')
    x = hist_pp.iloc[:, 1:] @ w
    stats = drawdown_stats(x, hist_pp.DATE)

    assert set(exp.columns) == set(stats.columns), "columns do not match"

    for c in stats.columns:
        if is_numeric_dtype(stats[c].dtype):
            assert_array_almost_equal(stats[c], exp[c], 5)
        else:
            assert stats[c].equals(exp[c]), f"{c} data are not equal"
