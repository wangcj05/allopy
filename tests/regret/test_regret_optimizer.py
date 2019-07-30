import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from allopy import OptData, RegretOptimizer
from allopy.datasets import load_monte_carlo
from .data import Test1, Test2
from .funcs import *


@pytest.fixture
def outlook():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data/scenario.csv"))


@pytest.mark.parametrize("config", [Test1, Test2])
def test_regret_optimizer(config, outlook):
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
    opt.optimize()

    assert_first_level_solution_equal_or_better(obj_funcs, opt.result.first_level_solutions, config.solutions)
    assert_almost_equal(opt.result.props, config.proportions, 3)


def assert_first_level_solution_equal_or_better(obj_funcs, solutions, expected):
    results = []
    for f, w, t, in zip(obj_funcs, solutions, expected):
        results.append(round(f(w) - f(t), 2) >= 0)

    assert np.alltrue(results)
