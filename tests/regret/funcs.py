import numpy as np
import pandas as pd

from allopy import OptData

__all__ = ["assets", "scenarios", "get_adjustments", "obj_max_returns", "cvar_fun", "sum_to_1"]

assets = 'DMEQ', 'EMEQ', 'PE', 'RE', 'NB', 'EILB', 'CASH'
scenarios = 'Baseline', 'Goldilocks', 'Stagflation', 'HHT'


def get_adjustments(outlook: pd.DataFrame, scenario: str, horizon: int, field: str):
    value_map = outlook.query(f"SCENARIO == '{scenario}' & HORIZON == {horizon}") \
        .groupby('ASSET') \
        .apply(lambda x: x[field]) \
        .droplevel(1)

    return [value_map[asset] for asset in assets]


def obj_max_returns(cube: OptData):
    def obj_fun(w):
        m = (cube @ w + 1).prod(0)
        ret = (np.sign(m) * np.abs(m) ** 0.05).mean() - 1
        return 1e2 * ret

    return obj_fun


def cvar_fun(cube: OptData, target: float):
    def cvar(w):
        return 1e3 * (target - cube.cvar(w, True, 5.0))

    return cvar


def sum_to_1(w):
    return sum(w) - 1
