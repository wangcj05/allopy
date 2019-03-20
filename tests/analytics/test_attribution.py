from numpy.testing import assert_array_almost_equal

from allopy.analytics.attribution import *


def test_cvar_attr(data, w, expected):
    msg = "cvar attribution does not match"
    cvar_data = data[:12]  # cut to 12 quarters.CVaR calculation usually uses 3 years

    attr = cvar_attr(cvar_data, w, False)
    assert_array_almost_equal(attr, expected["attr"]["cvar_no_rebal"], 4, f"No rebalance {msg}")

    attr = cvar_attr(cvar_data, w, True)
    assert_array_almost_equal(attr, expected["attr"]["cvar_rebal"], 4, f"Rebalance {msg}")


def test_returns_attr(cov, w, expected):
    attr = returns_attr(cov, w)
    assert_array_almost_equal(attr, expected["attr"]["returns"], 4)


def test_vol_attr(cov, w, expected):
    attr = risk_attr(cov, w)
    assert_array_almost_equal(attr, expected["attr"]["vol"], 4)
