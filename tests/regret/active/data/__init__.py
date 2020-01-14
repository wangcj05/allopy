import os

import pandas as pd

__all__ = ["adj", "cov_mat", "lb", "ub", "linear_constraints"]


def _file(fn: str):
    return os.path.join(os.path.dirname(__file__), fn)


adj = pd.read_csv(_file("adj.csv"))
cov_mat = pd.read_csv(_file("cov.csv")).iloc[:, 1:].to_numpy()
lb = pd.read_csv(_file("bounds.csv")).lb
ub = pd.read_csv(_file("bounds.csv")).ub
linear_constraints = pd.read_csv(_file("linear_constraints.csv"))
