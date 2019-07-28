import pickle
from itertools import product
from os import path
from typing import Dict, List, Tuple

import pandas as pd
import pytest
from copulae.core import corr2cov, near_psd

from allopy import OptData

__data_dir = path.join(path.dirname(__file__), 'data')


def get_excel_data(name: str, sheets: List[str] = None) -> Dict[str, pd.DataFrame]:
    if not name.lower().endswith('.xlsx'):
        name += '.xlsx'

    with pd.ExcelFile(path.join(__data_dir, name)) as xl:
        sheets = set(xl.sheet_names) if sheets is None else set(sheets)
        return {sheet: xl.parse(sheet) for sheet in xl.sheet_names if sheet in sheets}


def get_id(horizon: int, overweight: bool):
    return f"{horizon}Y{'O' if overweight else 'NO'}"


@pytest.fixture("module")
def bounds() -> Dict[str, Tuple[pd.Series, pd.Series]]:
    data = get_excel_data("meta-data", ['lb', 'ub'])
    return {col: (data['lb'][col], data['ub'][col]) for col in data['lb'].columns}


@pytest.fixture("module")
def ineq_cstr():
    A = get_excel_data("A")
    b = get_excel_data("meta-data", ["b"])["b"]

    return {col: (A[col], b[col]) for col in b.columns}


@pytest.fixture("module")
def exp_wgt():
    df = pd.read_excel(path.join(__data_dir, "meta-data.xlsx"), "exp_wgt", header=[0, 1])
    return {
        level: {
            col: df[level][col] for col in df[level].columns
        }
        for level in df.columns.levels[0]
    }


@pytest.fixture("module")
def sim_ret():
    sim_ret_map = {}
    meta_data = get_excel_data('meta-data', ['eva', 'risk_eva', 'passive_wgt', 'vol'])
    corr = get_excel_data('corr')

    for horizon, overweight in product([5, 10], [True, False]):
        with open(path.join(__data_dir, 'data.p'), 'rb') as f:
            orig = pickle.load(f)  # original data tensor

        id = get_id(horizon, overweight)
        cov = near_psd(corr2cov(corr[id], meta_data['vol'][id]))

        sim_ret_map[id] = (
            OptData(orig, 'quarterly')
                .set_cov_mat(cov)
                .cut_by_horizon(horizon)
                .calibrate_data(meta_data['eva'][id])
                .aggregate_assets(meta_data['passive_wgt'][id]),
            OptData(orig, 'quarterly')
                .set_cov_mat(cov)
                .cut_by_horizon(3)
                .calibrate_data(meta_data['risk_eva'][id])
                .aggregate_assets(meta_data['passive_wgt'][id])
        )

    return sim_ret_map
