import numpy as np

from allopy import ActivePortfolioRegretOptimizer
from .data import lb, linear_constraints, ub

probability = [0.5, 0.25, 0.25]


def test_active_regret_optimizer(main_cubes, cvar_cubes):
    opt = ActivePortfolioRegretOptimizer(main_cubes, cvar_cubes, probability)
    opt.set_bounds(lb, ub)
    opt = set_linear_constraints(opt)
    opt.maximize_info_ratio()

    results = opt.optimize()
    assert all(~np.isnan(results)), "should have non-nan results"


def set_linear_constraints(opt: ActivePortfolioRegretOptimizer):
    matrix = linear_constraints.iloc[:, :-2].to_numpy()
    B = linear_constraints.B.to_numpy()
    equalities = linear_constraints.EQUALITY.to_numpy()

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
