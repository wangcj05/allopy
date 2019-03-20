import numpy as np
import pandas as pd
import pickle
from os import path

__data_dir = path.join(path.dirname(__file__), '..', 'data')


def read_A(_id) -> np.ndarray:
    return pd.read_excel(path.join(__data_dir, 'A.xlsx'), _id).values


def read_corr(_id) -> np.ndarray:
    return pd.read_excel(path.join(__data_dir, 'corr.xlsx'), _id).values


def read_meta_data(_id):
    sheets = ['b', 'lb', 'ub', 'eva', 'risk_eva', 'passive_wgt', 'vol']
    fp = path.join(__data_dir, 'meta-data.xlsx')
    return [pd.read_excel(fp, sheet)[_id].values for sheet in sheets]


def get_exp_wgt(_id, target, ):
    return pd.read_excel(path.join(__data_dir, 'meta-data.xlsx'), 'exp_wgt', header=[0, 1])[target][_id]


def read_sim_ret() -> np.ndarray:
    with open(path.join(__data_dir, 'data.p'), 'rb') as f:
        return pickle.load(f)
