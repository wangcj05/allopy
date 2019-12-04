from typing import Dict

import pytest
from numpy.testing import assert_almost_equal

from allopy import ActivePortfolioOptimizer, OptData
from allopy.datasets import load_monte_carlo
from tests.active_portfolio.data import (
    get_adjustments, get_bounds, get_cov_mat, get_expected_results,
    get_linear_constraints, get_risk_cstr
)

scenarios = 'Baseline', 'Upside', 'Downside'


def parameters():
    adj = get_adjustments()
    risk = get_risk_cstr()
    results = get_expected_results()

    for scenario in scenarios:
        yield (
            scenario,
            adj[scenario].to_numpy(),
            float(risk.loc[0, scenario]),
            float(risk.loc[1, scenario]),
            results[scenario].to_numpy()
        )


def set_linear_constraints(opt: ActivePortfolioOptimizer):
    linear_constraints = get_linear_constraints()
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


@pytest.fixture(scope="module")
def scenario_data_map() -> Dict[str, OptData]:
    adj = get_adjustments()
    return {
        scenario: OptData(load_monte_carlo(total=True)).calibrate_data(adj[scenario], adj.Vol)
        for scenario in scenarios
    }


@pytest.fixture(scope="module")
def scenario_cvar_data_map() -> Dict[str, OptData]:
    adj = get_adjustments()

    return {
        scenario: OptData(load_monte_carlo(total=True))
            .cut_by_horizon(3)
            .calibrate_data(sd=adj.Vol)
        for scenario in scenarios
    }


@pytest.mark.parametrize("scenario, mean_adj, max_cvar, max_te, expected", parameters())
def test_optimizer(scenario_data_map, scenario_cvar_data_map, scenario, mean_adj, max_cvar, max_te, expected):
    agg_weights = [0.25, 0.18, 0.24, 0.13, 0.11, 0.04, 0.05]
    x0 = [1.0, 0.107, 0.008, 0.044, 0.064, 0.124, 0.009, 0.003, 0.003, 0.02, 0.006, 0.005, 0.054, 0.003, 0.0007,
          0.109, 0.004, 0.004, 0.01, 0.027, 0.001, 0.003, 0.003, 0.008, 0.006, 0.008, 0.002, 0.011, 0.0, 0.0]

    bounds = get_bounds()
    cube = scenario_data_map[scenario] \
        .alter_frequency('quarterly') \
        .aggregate_assets(agg_weights) \
        .set_cov_mat(get_cov_mat())

    cvar_cube = scenario_cvar_data_map[scenario] \
        .alter_frequency('quarterly') \
        .aggregate_assets(agg_weights)

    opt = ActivePortfolioOptimizer(cube, cvar_data=cvar_cube)
    opt = set_linear_constraints(opt)
    opt.set_bounds(bounds.lb, bounds.ub)
    results = opt.maximize_eva(max_te, max_cvar, x0=x0).round(4)
    assert_almost_equal(results, expected, 4)
