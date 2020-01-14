import os

import pandas as pd
import pytest

from allopy import OptData
from allopy.datasets import load_monte_carlo

_assets = 'DMEQ', 'EMEQ', 'PE', 'RE', 'NB', 'EILB', 'CASH'
_scenarios = 'Baseline', 'Goldilocks', 'Stagflation', 'HHT'


def get_adjustments(outlook: pd.DataFrame, scenario: str, horizon: int, field: str):
    value_map = outlook.query(f"SCENARIO == '{scenario}' & HORIZON == {horizon}") \
        .groupby('ASSET') \
        .apply(lambda x: x[field]) \
        .droplevel(1)

    return [value_map[asset] for asset in _assets]


@pytest.fixture("package")
def assets():
    return _assets


@pytest.fixture("package")
def scenarios():
    return _scenarios


@pytest.fixture("package")
def outlook():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data/scenario.csv"))


@pytest.fixture("package")
def main_cubes(outlook):
    return [
        OptData(load_monte_carlo()[..., :len(_assets)], 'quarterly')
            .calibrate_data(get_adjustments(outlook, scenario, 20, "RETURN"))
        for scenario in _scenarios
    ]


@pytest.fixture("package")
def cvar_cubes(outlook):
    return [
        OptData(load_monte_carlo()[..., :len(_assets)], 'quarterly')
            .cut_by_horizon(3)
            .calibrate_data(get_adjustments(outlook, scenario, 3, "RETURN"))
        for scenario in _scenarios
    ]
