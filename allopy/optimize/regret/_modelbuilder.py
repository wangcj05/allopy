import re
from typing import List

import nlopt as nl
import numpy as np

from allopy.optimize import BaseOptimizer
from allopy.optimize.algorithms import has_gradient, map_algorithm
from allopy.optimize.utils import create_matrix_constraint, validate_matrix_constraints


class ModelBuilder:
    def __init__(self,
                 num_assets: int,
                 num_scenarios: int,
                 algorithm,
                 obj_funcs,
                 lower_bounds,
                 upper_bounds,
                 eq_constraints,
                 ineq_constraints,
                 matrix_eq_constraints,
                 matrix_ineq_constraints,
                 sum_to_1: bool,
                 x_tol_abs: float,
                 x_tol_rel: float,
                 f_tol_abs: float,
                 f_tol_rel: float,
                 max_eval: float,
                 c_eps: float,
                 verbose: bool):
        algorithm = map_algorithm(algorithm) if isinstance(algorithm, str) else algorithm
        if has_gradient(algorithm) == 'NOT COMPILED':
            raise NotImplementedError(
                f"Cannot use '{nl.algorithm_name(algorithm)}' as it is not compiled")

        self.num_scenarios = num_scenarios
        self.num_assets = num_assets
        self.algorithm = algorithm
        self.max_or_min = None
        self._obj_funcs = obj_funcs
        self.lower_bounds = np.asarray(lower_bounds) if hasattr(lower_bounds, '__iter__') else np.repeat(0, num_assets)
        self.upper_bounds = np.asarray(upper_bounds) if hasattr(upper_bounds, '__iter__') else np.repeat(1, num_assets)
        self._meq = matrix_eq_constraints
        self._min = matrix_ineq_constraints
        self._heq = eq_constraints
        self._hin = ineq_constraints
        self.x_tol_abs = x_tol_abs
        self.x_tol_rel = x_tol_rel
        self.f_tol_abs = f_tol_abs
        self.f_tol_rel = f_tol_rel
        self.max_eval = max_eval
        self.c_eps = c_eps
        self.verbose = verbose
        self.sum_to_1 = sum_to_1

    def __call__(self, index: int):
        """Creates the individual optimization model for the first step"""
        assert isinstance(self.max_or_min, str)

        model = BaseOptimizer(self.num_assets, self.algorithm, verbose=self.verbose)
        model.set_bounds(self.lower_bounds, self.upper_bounds)

        # sets up optimizer's programs and bounds
        for item, set_option in [
            (self.x_tol_abs, model.set_xtol_abs),
            (self.x_tol_rel, model.set_xtol_rel),
            (self.f_tol_abs, model.set_ftol_abs),
            (self.f_tol_rel, model.set_ftol_rel),
            (self.max_eval, model.set_maxeval),
        ]:
            if item is not None:
                set_option(item)

        # sets constraints
        for constraints, set_constraint in [(self._min.values(), model.add_inequality_matrix_constraint),
                                            (self._meq.values(), model.add_equality_matrix_constraint)]:
            for c in constraints:
                set_constraint(c, self.c_eps)

        for constraints, set_constraint in [(self._heq.values(), model.add_equality_constraint),
                                            (self._hin.values(), model.add_inequality_constraint)]:
            for c in constraints:
                set_constraint(c[index], self.c_eps)

        if self.sum_to_1:
            model.add_equality_constraint(lambda x: sum(x) - 1)

        # sets up the objective function
        assert self.max_or_min in ('maximize', 'minimize') and len(self._obj_funcs) == self.num_scenarios, \
            "Objective function is not set yet. Use the .set_max_objective() or .set_min_objective() methods to do so"

        if self.max_or_min == "maximize":
            model.set_max_objective(self._obj_funcs[index])
        else:
            model.set_min_objective(self._obj_funcs[index])

        return model

    @property
    def heq(self):
        return self._meq

    @property
    def hin(self):
        return self._min

    @property
    def meq(self):
        return self._meq

    @property
    def min(self):
        return self._min

    @property
    def obj_func(self):
        return self._obj_funcs

    @obj_func.setter
    def obj_func(self, functions: list):
        self._validate_num_functions(functions)
        self._obj_funcs = functions

    def add_inequality_constraints(self, functions: list):
        self._validate_num_functions(functions)
        self._hin[self._build_name(functions[0].__name__, self._hin.keys())] = functions

    def add_equality_constraints(self, functions: list):
        self._validate_num_functions(functions)
        self._heq[self._build_name(functions[0].__name__, self._heq.keys())] = functions

    def add_inequality_matrix_constraints(self, A, b):
        A, b = validate_matrix_constraints(A, b)

        for i, row, _b in zip(range(len(b)), A, b):
            self._min[f'A_{i}'] = create_matrix_constraint(row, _b)

    def add_equality_matrix_constraints(self, Aeq, beq):
        Aeq, beq = validate_matrix_constraints(Aeq, beq)

        for i, row, _beq in zip(range(len(beq)), Aeq, beq):
            self._meq[f'Aeq_{i}'] = create_matrix_constraint(row, _beq)

    def _validate_num_functions(self, funcs: List):
        error_msg = f"Number of functions do not match. Functions given: {len(funcs)}. " \
                    f"Functions expected: {self.num_scenarios}"
        assert len(funcs) == self.num_scenarios, error_msg

    @staticmethod
    def _build_name(name, names):
        last_index = -1
        for n in names:
            match = re.match(name + r"(?:_(\d+))?$", n)
            if match:
                index = int(match.group(1) or 0)
                if index > last_index:
                    last_index = index

        return name if last_index == -1 else f"{name}_{last_index + 1}"
