import numpy as np
import pytest

from allopy import ActivePortfolioRegretOptimizer
from .data import lb, linear_constraints, ub

probability = [0.5, 0.25, 0.25]


def test_active_regret_optimizer_interface(main_cubes, cvar_cubes):
    opt = ActivePortfolioRegretOptimizer(main_cubes, cvar_cubes, probability, time_unit='quarter')
    opt.set_bounds(lb, ub)
    opt = set_linear_constraints(opt)
    results = opt.maximize_info_ratio()

    assert all(~np.isnan(results)), "should have non-nan results"


@pytest.mark.parametrize("dist_func", [np.abs, np.square])
def test_active_regret_optimizer_max_eva(main_cubes, cvar_cubes, dist_func):
    opt = ActivePortfolioRegretOptimizer(main_cubes, cvar_cubes, probability)
    opt.set_bounds(lb, ub)
    opt = set_linear_constraints(opt)
    sol = opt.maximize_eva(0.03, -0.4, dist_func=dist_func)
    assert all(~np.isnan(sol)), "should have non-nan results"


def set_linear_constraints(opt: ActivePortfolioRegretOptimizer):
    lc = linear_constraints.copy(True)
    matrix = lc.iloc[:, :-2].to_numpy()
    B = lc.B.to_numpy()
    equalities = lc.EQUALITY.to_numpy()

    # swap greater than or equal to equalities
    matrix[equalities == ">="] *= -1
    B[equalities == ">="] *= -1
    equalities[equalities == ">="] = "<="

    for eq in ["=", "<="]:
        a, b = matrix[equalities == eq], B[equalities == eq]
        if eq == "=":
            opt.add_equality_matrix_constraint(a, b)
        else:
            opt.add_inequality_matrix_constraint(a, b)

    return opt
