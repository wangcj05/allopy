import numpy as np
import pytest
from copulae.core import near_psd

from tests.analytics.setup import read_analytics
from tests.optimize.setup import read_sim_ret


@pytest.fixture("module")
def cov(data):
    cov_mat = np.mean([np.cov(data[:, i, :].T) for i in range(data.shape[1])], 0)
    return near_psd(cov_mat)


@pytest.fixture("module")
def data():
    return read_sim_ret()[:20, :, :7]


@pytest.fixture("module")
def expected():
    return read_analytics()


@pytest.fixture("module")
def w():
    return np.array([0.25, 0.18, 0.24, 0.13, 0.11, 0.04, 0.05])


@pytest.fixture("module")
def cw():
    return np.array([0.205, 0.175, 0.24, 0.13, 0.11, 0.04, 0.1])
