import pytest

from allopy import PortfolioRegretOptimizer
from allopy.optimize.regret.summary import RegretSummary
from .data import Test1


@pytest.mark.parametrize("config", [Test1])
def test_regret_optimizer_summary(config, assets, scenarios, main_cubes, cvar_cubes):
    opt = PortfolioRegretOptimizer(len(assets), len(scenarios), config.prob.as_array())
    opt.set_bounds(config.lb.as_array(), config.ub.as_array())

    opt.maximize_returns(main_cubes, cvar_cubes, max_cvar=config.cvar.as_array())
    opt.set_meta(asset_names=assets, scenario_names=scenarios)
    assert isinstance(opt.summary(), RegretSummary)
