import os
from concurrent.futures import ProcessPoolExecutor

import pytest

from allopy import OptData
from allopy.datasets import load_monte_carlo
from .data import assets, outlook, scenarios


def get_adjustments(scenario: str, horizon: int):
    returns_map = {r.ASSET: r.RETURN for _, r in
                   outlook.query(f"SCENARIO == '{scenario}' & HORIZON == {horizon}").iterrows()}

    return [returns_map[asset] for asset in assets]


@pytest.fixture("package")
def main_cubes():
    with ProcessPoolExecutor(os.cpu_count() - 1) as P:
        res = P.map(_derive_main_cube, scenarios)

    return list(res)


@pytest.fixture("package")
def cvar_cubes():
    with ProcessPoolExecutor(os.cpu_count() - 1) as P:
        res = P.map(_derive_cvar_cube, scenarios)

    return list(res)


def _derive_main_cube(scenario: str):
    mean = get_adjustments(scenario, 20)
    return OptData(load_monte_carlo(), 'quarterly') \
        .take_assets(len(assets)) \
        .calibrate_data(mean)


def _derive_cvar_cube(scenario: str):
    mean = get_adjustments(scenario, 3)
    return OptData(load_monte_carlo(), 'quarterly') \
        .take_assets(len(assets)) \
        .cut_by_horizon(3) \
        .calibrate_data(mean)
