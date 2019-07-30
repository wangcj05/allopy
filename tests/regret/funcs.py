import numpy as np

from allopy import OptData

__all__ = ["obj_max_returns", "cvar_fun", "sum_to_1"]


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
