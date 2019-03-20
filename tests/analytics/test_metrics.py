from numpy.testing import assert_almost_equal

from allopy.analytics.metrics import *

weights = 0.4, 0.6


def test_risk_metrics(data, w, expected):
    msg = "risk metric does not match"
    fields = 'cvar', 'tail_risk', 'sstd'

    cvar_data = data.copy()[:12]
    expected = expected['metrics']['risk']

    for rebal in (False, True):
        ind = 'rebal' if rebal else 'no_rebal'
        exp = expected.loc[ind]

        res = risk_metrics(data, w, rebal, cvar_data=cvar_data)

        for f in fields:
            sub_msg = 'Rebalance' if rebal else 'No Rebalance'
            assert_almost_equal(res[f], exp[f], 4, f"{sub_msg} - {f} - {msg}")


def test_returns_metrics(data, w, cw, cov, expected):
    msg = "returns metric does not match"
    fields = 'mean', 'std', 'skew', 'kurtosis'

    expected = expected['metrics']['returns']

    for rebal in (False, True):
        ind = 'rebal' if rebal else 'no_rebal'
        exp = expected.loc[ind]

        res = returns_metrics(data, w, cov, rebal, 4, cw, weights)

        for f in fields:
            sub_msg = 'Rebalance' if rebal else 'No Rebalance'
            assert_almost_equal(res[f], exp[f], 4, f"{sub_msg} - {f} - {msg}")


def test_ratios(data, w, cw, cov, expected):
    msg = "performance ratio does not match"
    fields = 'sharpe', 'sortino', 'trasr'

    expected = expected['metrics']['ratios']
    cvar_data = data.copy()[:12]

    for rebal in (False, True):
        ind = 'rebal' if rebal else 'no_rebal'
        exp = expected.loc[ind]

        returns_mx = returns_metrics(data, w, cov, rebal, 4, cw, weights)
        risk_mx = risk_metrics(data, w, rebal, cvar_data=cvar_data)
        pr_ = performance_ratios_(data, w, cov, rebal, 4, cw, weights, cvar_data=cvar_data)
        pr = performance_ratios(returns_mx, risk_mx)

        for f in fields:
            sub_msg = 'Rebalance' if rebal else 'No Rebalance'
            assert_almost_equal(pr_[f], exp[f], 4, f"{sub_msg} - {f} - {msg}")
            assert_almost_equal(pr[f], exp[f], 4, f"{sub_msg} - {f} - {msg}")
