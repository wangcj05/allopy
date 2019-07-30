import pytest

from allopy import OptData, RegretOptimizer
from allopy.datasets import load_monte_carlo
from .data import Test1
from .funcs import *


@pytest.mark.parametrize("config", [Test1])
def test_regret_optimizer_summary(config, outlook):
    opt = RegretOptimizer(len(assets), len(scenarios), config.prob.as_array())
    opt.set_bounds(config.lb.as_array(), config.ub.as_array())

    obj_funcs = []
    constraint_funcs = []
    for scenario in scenarios:
        main = OptData(load_monte_carlo()[..., :len(assets)], 'quarterly') \
            .calibrate_data(get_adjustments(outlook, scenario, 20, 'RETURN'))

        cvar = OptData(load_monte_carlo()[..., :len(assets)], 'quarterly') \
            .cut_by_horizon(3) \
            .calibrate_data(get_adjustments(outlook, scenario, 3, 'RETURN'))

        obj_funcs.append(obj_max_returns(main))
        constraint_funcs.append(cvar_fun(cvar, config.cvar[scenario]))

    opt.set_max_objective(obj_funcs)
    opt.add_inequality_constraint(constraint_funcs)
    opt.add_equality_constraint([sum_to_1] * len(scenarios))
    opt.optimize(config.solutions, config.proportions)

    opt.set_meta(asset_names=assets, scenario_names=scenarios)
    assert opt.summary() is not None
