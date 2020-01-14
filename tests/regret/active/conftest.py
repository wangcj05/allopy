from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pytest

from allopy import OptData
from allopy.datasets import load_monte_carlo
from .data import adj, cov_mat

# constants
scenarios = 'Baseline', 'Upside', 'Downside'
agg_weights = [0.25, 0.18, 0.24, 0.13, 0.11, 0.04, 0.05]


@pytest.fixture("package")
def main_cubes():
    with ProcessPoolExecutor(max(cpu_count() - 1, 1)) as P:
        res = P.map(_derive_main_cube, scenarios)

    return list(res)


def _derive_main_cube(scenario: str):
    cube = load_monte_carlo(total=True)

    return OptData(cube, 'monthly') \
        .calibrate_data(mean=adj[scenario], sd=adj.Vol) \
        .alter_frequency('quarterly') \
        .aggregate_assets(agg_weights) \
        .set_cov_mat(cov_mat)


@pytest.fixture("package")
def cvar_cubes():
    cube = load_monte_carlo(total=True)

    return [OptData(cube, 'monthly')
                .cut_by_horizon(3)
                .calibrate_data(sd=adj.Vol)
                .alter_frequency('quarterly')
                .aggregate_assets(agg_weights)] * len(scenarios)
