"""
This file tests AP optimization. By default, AP optimization does not rebalance [REBALANCE = False]

We test the optimization regimes on 5 and 10 years with overweight set to False and True for a total of 4 combinations.
Overweight refers to whether 2 additional asset classes are added.

We have 9 ASR optimization programs:
1) Maximize EVA st Risk and CVaR constraints
2) Maximize EVA st Risk constraint
3) Maximize EVA st CVaR constraint
4) Minimize Tracking Error
5) Minimize CVaR
5) Minimize CVaR st active return constraint
1) Maximize Information Ratio

Notes:
    Most test start by setting a random seed at the start. This is because unless we supply a initial vector, the
    initial vector will be randomly generated. And there is no guarantee that the optimizer will find a solution with
    the random vector generated
"""

import numpy as np
import pytest
from copulae.core import corr2cov, near_psd
from functools import lru_cache
from nlopt import RoundoffLimited

from allopy.optimize import ASROptimizer, OptData
from tests.optimize.setup import get_exp_wgt, read_A, read_corr, read_meta_data, read_sim_ret

decor_horizon = pytest.mark.parametrize('horizon', [5, 10])
decor_overweight = pytest.mark.parametrize('overweight', [False, True])

REBALANCE = False
EPS = 1e-6


def assert_equal_or_better_solution(obj_fn, sol, expected, name, bigger_better, constraints=None):
    """
    Convenience assertion function. This function will pass if any of the conditions are met

    1) Solution provided is valid and is better (than the sample we obtained from R)
    2) Solution weights is close to expected weights
    3) Solution objective values are is close to expected objective weights

    :param obj_fn: callable
        objective function
    :param sol: ndarray
        solution weights
    :param expected: ndarray
        sample weight
    :param name: str
        name of the test
    :param bigger_better: bool
        Set this to True if a bigger value for the objective function is better (max objective), otherwise False
    :param constraints: List[Callable[ndarray, bool]]
        list of callable functions which takes in a single weight parameter and returns True or False depending on
        whether the constraint was violated or not
    """

    tol = 1e-3  # 1% difference at most
    sol_value, exp_value = obj_fn(sol), obj_fn(expected)

    better_solution = bigger_better == (sol_value > exp_value)  # exclusive and operation
    if constraints is not None:
        for cons in constraints:
            better_solution &= cons(sol)

    weights_are_close = np.allclose(sol, expected, atol=tol)  # 0.1% difference in weight
    func_values_are_close = np.isclose(sol_value, exp_value, rtol=tol)  # 0.1% relative difference in objective func

    assert better_solution or weights_are_close or func_values_are_close, \
        f"{name}: Solution is not better and neither weights nor function values are close. Relative tolerance: {tol}"


def cvar(data):
    def func(w):
        return data.cvar(w, REBALANCE)

    return func


def tracking_error(data):
    def func(w):
        w = [0, *w[1:]]
        return data.volatility(w)

    return func


def active_return(data):
    def func(w):
        w = [0, *w[1:]]
        return data.expected_return(w, REBALANCE)

    return func


def total_return(data):
    def func(w):
        return data.expected_return(w, REBALANCE)

    return func


@lru_cache()
def get_opt_data(horizon, overweight):
    _id = f"{horizon}Y{'O' if overweight else 'NO'}"
    orig = read_sim_ret()

    A = read_A(_id)
    b, lb, ub, eva, risk_eva, passive_wgt, vol = read_meta_data(_id)
    corr = read_corr(_id)
    cov = near_psd(corr2cov(corr, vol))

    data = OptData(orig, cov, 4).cut_by_horizon(horizon).calibrate_data(eva).aggregate_assets(passive_wgt)
    risk_data = OptData(orig, cov, 4).cut_by_horizon(3).calibrate_data(risk_eva).aggregate_assets(passive_wgt)

    return _id, data, risk_data, A, b, lb, ub


@decor_horizon
@decor_overweight
def test_max_eva_st_te_cvar(horizon, overweight):
    np.random.seed(8888)
    target = 'MAX_EVA_ST_TE_CVAR'

    _id, data, risk_data, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    opt = (ASROptimizer(data, cvar_data=risk_data, rebalance=REBALANCE)
           .set_bounds(lb, ub)
           .add_inequality_matrix_constraint(A, b))

    max_te = 0.03
    max_cvar = -0.4

    sol = opt.AP.max_eva_st_risk(max_te, max_cvar)

    assert_equal_or_better_solution(total_return(data), sol, exp_wgt, f'{target} {_id}', bigger_better=True,
                                    constraints=[
                                        (lambda x: tracking_error(data)(x) <= max_te + EPS),
                                        (lambda x: cvar(data)(x) >= max_cvar - EPS)
                                    ])


@decor_horizon
@decor_overweight
def test_max_eva_st_te(horizon, overweight):
    np.random.seed(8888)
    target = 'MAX_EVA_ST_TE'

    _id, data, risk_data, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    opt = (ASROptimizer(data, cvar_data=risk_data, rebalance=REBALANCE)
           .set_bounds(lb, ub)
           .add_inequality_matrix_constraint(A, b))

    max_te = 0.03

    sol = opt.AP.max_eva_st_risk(max_te)

    assert_equal_or_better_solution(total_return(data), sol, exp_wgt, f'{target} {_id}', bigger_better=True,
                                    constraints=[
                                        (lambda x: tracking_error(data)(x) <= max_te + EPS)
                                    ])


@decor_horizon
@decor_overweight
def test_max_eva_st_cvar(horizon, overweight):
    np.random.seed(8888)
    target = 'MAX_EVA_ST_CVAR'

    _id, data, risk_data, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    opt = (ASROptimizer(data, cvar_data=risk_data, rebalance=REBALANCE)
           .set_bounds(lb, ub)
           .add_inequality_matrix_constraint(A, b))

    max_cvar = -0.4

    sol = opt.AP.max_eva_st_risk(max_cvar=max_cvar)

    assert_equal_or_better_solution(total_return(data), sol, exp_wgt, f'{target} {_id}', bigger_better=True,
                                    constraints=[
                                        (lambda x: cvar(data)(x) >= max_cvar - EPS)
                                    ])


@decor_horizon
@decor_overweight
def test_min_te(horizon, overweight):
    np.random.seed(8888)
    target = 'MIN_TE'

    _id, data, _, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    opt = (ASROptimizer(data, rebalance=REBALANCE).set_bounds(lb, ub).add_inequality_matrix_constraint(A, b))
    sol = opt.AP.min_tracking_error()

    assert_equal_or_better_solution(tracking_error(data), sol, exp_wgt, f'{target} {_id}', bigger_better=False)


@decor_horizon
@decor_overweight
def test_min_te_st_active_ret(horizon, overweight):
    np.random.seed(8888)
    target = 'MIN_TE_ST_ACTIVE_RET'

    _id, data, _, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    min_ret = 0.01

    opt = (ASROptimizer(data, rebalance=REBALANCE).set_bounds(lb, ub).add_inequality_matrix_constraint(A, b))
    sol = opt.AP.min_tracking_error(min_ret, True)

    assert_equal_or_better_solution(tracking_error(data), sol, exp_wgt, f'{target} {_id}', bigger_better=False,
                                    constraints=[
                                        lambda x: active_return(data)(x) >= min_ret - EPS
                                    ])


@decor_horizon
@decor_overweight
def test_min_cvar(horizon, overweight):
    np.random.seed(8888)
    target = 'MIN_CVAR'

    _id, data, _, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    while True:
        try:
            opt = (ASROptimizer(data, rebalance=REBALANCE).set_bounds(lb, ub).add_inequality_matrix_constraint(A, b))
            sol = opt.AP.min_cvar()
            break
        except RoundoffLimited:
            pass

    assert_equal_or_better_solution(cvar(data), sol, exp_wgt, f'{target} {_id}', bigger_better=True)


@decor_horizon
@decor_overweight
def test_min_cvar_st_active_ret(horizon, overweight):
    np.random.seed(8888)
    target = 'MIN_CVAR_ST_ACTIVE_RET'

    _id, data, _, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    min_ret = 0.01

    while True:
        try:
            opt = (ASROptimizer(data, rebalance=REBALANCE).set_bounds(lb, ub).add_inequality_matrix_constraint(A, b))
            sol = opt.AP.min_cvar(min_ret, True)
            break
        except RoundoffLimited:
            pass

    assert_equal_or_better_solution(cvar(data), sol, exp_wgt, f'{target} {_id}', bigger_better=True,
                                    constraints=[
                                        lambda x: active_return(data)(x) >= min_ret - EPS
                                    ])


@decor_horizon
@decor_overweight
def test_max_information_ratio(horizon, overweight):
    np.random.seed(8888)
    target = 'MAX_IR'

    _id, data, _, A, b, lb, ub = get_opt_data(horizon, overweight)
    exp_wgt = get_exp_wgt(_id, target)

    opt = (ASROptimizer(data, rebalance=REBALANCE).set_bounds(lb, ub).add_inequality_matrix_constraint(A, b))
    sol = opt.AP.max_info_ratio()

    def func(x):
        x = [0, *x[1:]]
        return data.sharpe_ratio(x, REBALANCE)

    assert_equal_or_better_solution(func, sol, exp_wgt, f'{target} {_id}', bigger_better=True)
