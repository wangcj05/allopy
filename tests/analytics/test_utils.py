from numpy.testing import assert_almost_equal

from allopy.analytics.utils import weighted_annualized_returns


def test_weighted_returns(data, w, cw):
    ann_ret = weighted_annualized_returns(data, False, 4, (0.6, cw), (0.4, w))
    assert_almost_equal(ann_ret, 0.0463385, 4)

    ann_ret = weighted_annualized_returns(data, True, 4, (0.6, cw), (0.4, w))

    assert_almost_equal(ann_ret, 0.0437894, 4)
