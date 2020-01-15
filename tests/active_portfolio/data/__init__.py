import os

import pandas as pd


def read_csv(name: str) -> pd.DataFrame:
    if not name.endswith('.csv'):
        name += '.csv'
    file = os.path.join(os.path.dirname(__file__), name)
    return pd.read_csv(file)


adj = read_csv("adj")
cov_mat = read_csv("cov").iloc[:, 1:].to_numpy()
lb = read_csv("bounds").lb.to_numpy()
ub = read_csv("bounds").ub.to_numpy()


def get_expected_results():
    return read_csv("results.csv")


def get_linear_constraints():
    return read_csv("linear_constraints")


def get_risk_cstr():
    return read_csv('risk')
