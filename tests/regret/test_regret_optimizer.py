import numpy as np
import pytest

from allopy import PortfolioRegretOptimizer, RegretOptimizer, get_option
from .data import Test1, Test2
from .funcs import cvar_fun, obj_max_returns


@pytest.mark.parametrize("config", [Test1, Test2])
def test_regret_optimizer(config, assets, scenarios, main_cubes, cvar_cubes):
    opt = RegretOptimizer(len(assets), len(scenarios), config.prob.as_array(), sum_to_1=True)
    opt.set_bounds(config.lb.as_array(), config.ub.as_array())

    obj_funcs, constraint_funcs = [], []
    for i, scenario in enumerate(scenarios):
        obj_funcs.append(obj_max_returns(main_cubes[i]))
        constraint_funcs.append(cvar_fun(cvar_cubes[i], config.cvar[scenario]))

    opt.set_max_objective(obj_funcs)
    opt.add_inequality_constraint(constraint_funcs)
    opt.optimize()

    assert_scenario_solution_equal_or_better(obj_funcs, opt.result.scenario_solutions, config.solutions)
    # assert_almost_equal(opt.result.proportions, config.proportions, 3)


@pytest.mark.parametrize("config", [Test1])
def test_portfolio_regret_optimizer(config, assets, scenarios, main_cubes, cvar_cubes):
    opt = PortfolioRegretOptimizer(main_cubes, cvar_cubes, config.prob.as_array(),
                                   rebalance=True, sum_to_1=True, time_unit='quarterly')

    opt.set_bounds(config.lb.as_array(), config.ub.as_array())
    opt.maximize_returns(max_cvar=config.cvar.as_array())

    obj_funcs = opt._objectives.max_returns
    assert_scenario_solution_equal_or_better(obj_funcs, opt.solution.scenario_optimal, config.solutions)
    assert_regret_is_lower(opt.solution.proportions,
                           config.proportions,
                           opt.solution.scenario_optimal,
                           obj_funcs,
                           config.prob.as_array())


def assert_scenario_solution_equal_or_better(obj_funcs, solutions, expected):
    results = []
    for f, w, t, in zip(obj_funcs, solutions, expected):
        diff = (f(w) - f(t)) / get_option("F.SCALE")
        results.append(round(diff, 4) >= 0)

    assert np.alltrue(results)


def assert_regret_is_lower(p0, p1, solutions, obj_funcs, prob):
    def regret(p):
        f_values = np.array([obj_funcs[i](s) for i, s in enumerate(solutions)])
        cost = f_values - np.array([f(p @ solutions) for f in obj_funcs])
        cost = np.asarray(cost ** 2)
        return 100 * sum(prob * cost)

    diff = (regret(p0) - regret(p1)) / get_option("F.SCALE")
    assert round(diff) <= 0
