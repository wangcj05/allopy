import inspect
from collections import abc
from itertools import zip_longest
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import nlopt as nl
import numpy as np
import numpy.random as rng
from copulae.types import Numeric

from .algorithms import LD_SLSQP, _has_gradient, _map_algorithm

__all__ = ['BaseOptimizer']
ConstraintMap = Dict[str, Callable]


class BaseOptimizer:
    _A: np.ndarray
    _b: np.ndarray
    _Aeq: np.ndarray
    _beq: np.ndarray

    def __init__(self, n: int, algorithm=LD_SLSQP, eps=np.finfo('float').eps ** (1 / 3), *args, **kwargs):
        """
        The BaseOptimizer is the raw optimizer with minimal support. For advanced users, this class will provide
        the most flexibility. The default algorithm used is Sequential Least Squares Quadratic Programming.

        Parameters
        ----------
        n: int
            number of assets

        algorithm: str
            the optimization algorithm

        eps: float
            the gradient step

        args:
            other arguments to setup the optimizer

        kwargs:
            other keyword arguments
        """
        if isinstance(algorithm, str):
            algorithm = _map_algorithm(algorithm)

        self._n = n
        self._model = nl.opt(algorithm, n, *args)

        has_grad = _has_gradient(algorithm)
        if has_grad == 'NOT COMPILED':
            raise NotImplementedError(f"Cannot use '{nl.algorithm_name(algorithm)}' as it is not compiled")

        self._auto_grad: bool = kwargs.get('auto_grad', has_grad)
        self._eps = eps
        self.set_xtol_abs(1e-5)
        self.set_xtol_rel(1e-3)
        self.set_maxeval(1000)
        self.set_ftol_rel(0)
        self.set_ftol_abs(0)
        self._hin: ConstraintMap = {}
        self._heq: ConstraintMap = {}
        self._result: Result = None
        self._max_or_min = None

    @property
    def lower_bounds(self):
        """Lower bound of each variable"""
        return np.asarray(self._model.get_lower_bounds(), np.float64)

    @lower_bounds.setter
    def lower_bounds(self, lb: Union[int, float, np.ndarray]):
        self.set_lower_bounds(lb)

    @property
    def upper_bounds(self):
        """Upper bound of each variable"""
        return np.asarray(self._model.get_upper_bounds(), np.float64)

    @upper_bounds.setter
    def upper_bounds(self, ub: Union[int, float, np.ndarray]):
        self.set_upper_bounds(ub)

    def optimize(self, x0: Optional[Iterable[float]] = None, *args) -> np.ndarray:
        """
        Runs the optimizer. If no initial vector is set, the optimizer will randomly generate a feasible one.

        Parameters
        ----------
        x0: iterable float
            Initial vector. Starting position for free variables

        args:
            other arguments to pass into the optimizer

        Returns
        -------
        ndarray
            Values of free variables at optimality
        """
        if x0 is None:
            lb = self.lower_bounds
            ub = self.upper_bounds

            if np.isinf(lb).any() or np.isinf(ub).any():
                x0 = rng.uniform(size=self._n)
            else:
                x0 = rng.uniform(lb, ub)
        else:
            x0 = np.asarray(x0)

        sol = self._model.optimize(x0, *args)
        self._result = Result(self._hin, self._heq, sol, self._eps)

        return self._result.x

    def _get_gradient_func(self, fn: Callable):
        if self._auto_grad and len(inspect.signature(fn).parameters) == 1:
            print(f"Setting gradient for function: '{fn.__name__}'")
            return _create_gradient_func(fn, self._eps)
        else:
            return fn

    def set_max_objective(self, fn: Callable, *args):
        """
        Sets the optimizer to maximize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        Parameters
        ----------
        fn: Callable
            Objective function

        args:
            Other arguments to pass to the objective function. This can be ignored in most cases

        Returns
        -------
        BaseOptimizer
            Own instance
        """
        self._max_or_min = 'maximize'
        self._model.set_stopval(float('inf'))

        f = self._get_gradient_func(fn)
        self._model.set_max_objective(f, *args)

        return self

    def set_min_objective(self, fn: Callable, *args):
        """
        Sets the optimizer to minimize the objective function. If gradient of the objective function is not set and the
        algorithm used to optimize is gradient-based, the optimizer will attempt to insert a smart numerical gradient
        for it.

        Parameters
        ----------
        fn: Callable
            Objective function

        args:
            Other arguments to pass to the objective function. This can be ignored in most cases
        """
        self._max_or_min = 'minimize'
        self._model.set_stopval(-float('inf'))
        f = self._get_gradient_func(fn)
        self._model.set_min_objective(f, *args)

        return self

    def add_inequality_constraint(self, fn: Callable, *args):
        """
        Adds the equality constraint function in standard form, A <= b. If the gradient of the constraint function is
        not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        Parameters
        ----------
        fn: Callable
            Constraint function

        args:
            Other arguments to pass to the constraint function. This can be ignored in most cases
        """
        f = self._get_gradient_func(fn)
        self._hin[fn.__name__] = fn
        self._model.add_inequality_constraint(f, *args)
        return self

    def add_equality_constraint(self, fn: Callable, *args):
        """
        Adds the equality constraint function in standard form, A = b. If the gradient of the constraint function
        is not specified and the algorithm used is a gradient-based one, the optimizer will attempt to insert a smart
        numerical gradient for it.

        Parameters
        ----------
        fn: Callable
            Constraint function

        args:
            Other arguments to pass to the constraint function. This can be ignored in most cases
        """
        f = self._get_gradient_func(fn)
        self._heq[fn.__name__] = fn
        self._model.add_equality_constraint(f, *args)
        return self

    def add_inequality_matrix_constraint(self, A, b):
        r"""
        Sets inequality constraints in standard matrix form.

        For inequality, :math:`\mathbf{A} \cdot \mathbf{x} \leq \mathbf{b}`

        Parameters
        ----------
        A: {iterable float, ndarray}
            Inequality matrix. Must be 2 dimensional.

        b: {scalar, ndarray}
            Inequality vector or scalar. If scalar, it will be propagated.
        """
        A, b = _validate_matrix_constraints(A, b)

        for i, row, _b in zip(range(len(b)), A, b):
            fn = _create_matrix_constraint(row, _b)
            f = _create_gradient_func(fn, self._eps)
            self._model.add_inequality_constraint(f)
            self._hin[f'A_{i}'] = fn

        return self

    def add_equality_matrix_constraint(self, Aeq, beq):
        r"""
        Sets equality constraints in standard matrix form.

        For equality, :math:`\mathbf{A} \cdot \mathbf{x} = \mathbf{b}`

        Parameters
        ----------
        Aeq: {iterable float, ndarray}
            Equality matrix. Must be 2 dimensional.

        beq: {scalar, ndarray}
            Equality vector or scalar. If scalar, it will be propagated.ndarray
        """
        Aeq, beq = _validate_matrix_constraints(Aeq, beq)

        for i, row, _beq in zip(range(len(beq)), Aeq, beq):
            fn = _create_matrix_constraint(row, _beq)
            f = _create_gradient_func(fn, self._eps)
            self._model.add_equality_constraint(f)
            self._heq[f'Aeq_{i}'] = fn

        return self

    def remove_all_constraints(self):
        """Removes all constraints"""
        self.remove_equality_constraints()
        self.remove_equality_constraints()
        return self

    def remove_inequality_constraints(self):
        """Removes all inequality constraints"""
        self._model.remove_inequality_constraints()
        return self

    def remove_equality_constraints(self):
        """Removes all equality constraints"""
        self._model.remove_equality_constraints()
        return self

    def set_bounds(self, lb: Numeric, ub: Numeric):
        """
        Sets the lower and upper bound

        Parameters
        ----------
        lb: {int, float, ndarray}
            Vector of lower bounds. If array, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        ub: {int, float, ndarray}
            Vector of upper bounds. If array, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.

        Returns
        -------
        BaseOptimizer
            Own instance
        """
        self.set_lower_bounds(lb)
        self.set_upper_bounds(ub)

        return self

    def set_lower_bounds(self, lb: Numeric):
        """
        Sets the lower bounds

        Parameters
        ----------
        lb: {int, float, ndarray}
            Vector of lower bounds. If vector, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.
        """
        if isinstance(lb, (int, float)):
            lb = np.repeat(float(lb), self._n)

        assert len(lb) == self._n, f"Input vector length must be {self._n}"

        self._model.set_lower_bounds(np.asarray(lb))
        return self

    def set_upper_bounds(self, ub: Numeric):
        """
        Sets the upper bound

        Parameters
        ----------
        ub: {int, float, ndarray}
            Vector of lower bounds. If vector, must be same length as number of free variables. If :class:`float` or
            :class:`int`, value will be propagated to all variables.
        """
        if isinstance(ub, (int, float)):
            ub = np.repeat(float(ub), self._n)

        assert len(ub) == self._n, f"Input vector length must be {self._n}"

        self._model.set_upper_bounds(np.asarray(ub))
        return self

    def set_epsilon(self, epsilon: float):
        """
        Sets the step difference used when calculating the gradient for derivative based optimization algorithms.
        This can ignored if you use a derivative free algorithm or if you specify your gradient specifically.

        Parameters
        ----------
        epsilon: float
            the gradient step
        """
        self._eps = epsilon
        return self

    def set_xtol_abs(self, tol: Union[float, np.ndarray]):
        """
        Sets absolute tolerances on optimization parameters.

        The tol input must be an array of length `n` specified in the initialization. Alternatively, pass a single
        number in order to set the same tolerance for all optimization parameters.

        Parameters
        ----------
        tol: {float, ndarray}
            Absolute tolerance for each of the free variables
        """
        if not isinstance(tol, float):
            tol = np.asarray(tol, dtype=float)
        self._model.set_xtol_abs(tol)
        return self

    def set_xtol_rel(self, tol: Union[float, np.ndarray]):
        """
        Sets relative tolerances on optimization parameters.

        The tol input must be an array of length `n` specified in the initialization. Alternatively, pass a single
        number in order to set the same tolerance for all optimization parameters.

        Parameters
        ----------
        tol: {float, ndarray}
            relative tolerance for each of the free variables
        """
        if not isinstance(tol, float):
            tol = np.asarray(tol, dtype=float)
        self._model.set_xtol_rel(tol)
        return self

    def set_maxeval(self, n: int):
        """
        Sets maximum number of objective function evaluations.

        After maximum number of evaluations, optimization will stop. Set 0 or negative for no limit.

        Parameters
        ----------
        n: int
            maximum number of evaluations
        """
        self._model.set_maxeval(n)
        return self

    def set_ftol_abs(self, tol: float):
        """
        Set absolute tolerance on objective function value

        Parameters
        ----------
        tol: float
            absolute tolerance of objective function value
        """
        self._model.set_ftol_abs(tol)
        return self

    def set_ftol_rel(self, tol: float):
        """
        Set relative tolerance on objective function value

        Parameters
        ----------
        tol: float
            Absolute relative of objective function value
        """
        self._model.set_ftol_rel(tol)
        return self

    def set_stopval(self, stopval: float):
        """
        Stop when an objective value of at least/most stopval is found depending on min or max objective

        Parameters
        ----------
        stopval: float
            Stopping value
        """
        self._model.set_stopval(stopval)
        return self

    def summary(self):
        """Prints a summary report of the optimizer"""

        model = self._model

        prog_setup = [
            ('objective', self._max_or_min),
            ('n_var', self._n),
            ('n_eq_con', len(self._heq)),
            ('n_ineq_con', len(self._hin)),
        ]

        opt_setup = [
            ('xtol_abs', model.get_xtol_abs()[0]),
            ('xtol_rel', model.get_xtol_rel()),
            ('ftol_abs', model.get_ftol_abs()),
            ('ftol_rel', model.get_ftol_rel()),
            ('max_eval', model.get_maxeval()),
            ('stop_val', model.get_stopval()),
        ]

        bounds = self.lower_bounds, self.upper_bounds
        smry = Summary(model.get_algorithm_name(), prog_setup, opt_setup, bounds)

        r = self._result
        if r is not None:
            smry.tight_hin = r.tight_hin
            smry.violations = r.violations
            smry.solution = r.x
        return smry


class Result:
    def __init__(self, hin: ConstraintMap, heq: ConstraintMap, sol: np.ndarray, eps: float):
        self.tight_hin = []
        self.violations = []

        for name, f in hin.items():
            value = f(sol)
            if -eps <= value <= eps:
                self.tight_hin.append(name)
            elif value > eps:
                self.violations.append(name)

        for name, f in heq.items():
            value = f(sol)
            if abs(value) > eps:
                self.violations.append(name)

        if len(self.violations) == 0:
            self.x = sol
        else:
            print('No solution was found for the given problem. Check the summary() for more information')
            self.x = None


class Summary:
    def __init__(self, algorithm: str, prog_setup: list, opt_setup: list, bounds: Tuple[np.ndarray, np.ndarray]):
        self.alg = algorithm
        self.prog_setup = prog_setup
        self.opt_setup = opt_setup
        self.violations = []
        self.tight_hin = []
        self.solution = []
        self.lb, self.ub = bounds

    def _repr_html_(self):
        """Used when printing to IPython notebooks"""

        def wrap(tag, element):
            return f"<{tag}>{element}</{tag}>"

        setup = ""
        for l1, l2 in zip_longest(self.prog_setup, self.opt_setup):
            x1, x2 = ('', '') if l1 is None else l1
            y1, y2 = ('', '') if l2 is None else l2

            setup += wrap('tr', ''.join(wrap('td', i) for i in (x1, x2, y1, y2)))

        bounds = " ".join(
            wrap(
                'tr',
                ''.join([wrap('td', round(l, 6)), wrap('td', round(6))]))
            for l, u in zip(self.lb, self.ub))

        if len(self.violations) > 0:
            violations = ''.join(wrap('li', f"{i + 1:3d}: {n}") for i, n in enumerate(self.violations))
            results = f"""
<div>
    <b>No solution found</b>. List of constraints violated below:
    <ul>
    {violations}
    </ul>
</div>
            """
        else:
            if len(self.tight_hin) == 0:
                tight = 'None of the constraints were tight'
            else:
                tight = ''.join(wrap("li", f'{i + 1:3d}: {n}') for i, n in enumerate(self.tight_hin))
                tight = f'The following inequality constraints were tight: <br/><ul>{tight}</ul>'

            sol = self.solution
            results = f"""
<div>
    <b>Program found a solution</b>
    <p>
        Solution: [{', '.join(str(round(x, 6)) for x in sol) if isinstance(sol, abc.Iterable) else sol}]
    </p>
    {tight}
</div> 
"""

        return f"""
<h1>GIC Portfolio Optimizer</h1>
<h3>Algorithm: {self.alg}</h3>
<hr/>
<table>
    <tr>
        <th>Problem Setup</th>
        <th>Value</th>
        <th>Optimizer Setup</th>
        <th>Value</th>
    </tr>
    {setup}
</table>
<hr/>
<table>
    <tr>
        <th>Lower Bound</th>
        <th>Upper Bound</th>
    </tr>
    {bounds}
</table>
<hr/>
<h3>Results</h3>
{results}
        """

    def as_text(self):
        n = 84

        def divider(char='-'):
            return '\n'.join(['', char * n, ''])

        def new_lines(x=1):
            return '\n' * x

        # names
        rows = [
            f"{'GIC Portfolio Optimizer':^84s}",
            divider('='),
            new_lines(),
            f'Algorithm: {self.alg}',
            divider(),
            f'{"Optimizer Setup":42s}{"Options":42s}',
        ]

        # optimization details
        for l1, l2 in zip_longest(self.prog_setup, self.opt_setup):
            x1, x2 = ('', '') if l1 is None else l1
            y1, y2 = ('', '') if l2 is None else l2
            rows.append(f"{x1:28s}{str(x2):>12s}    {y1:28s}{str(y2):>12s}")

        # bounds
        rows.extend([
            divider(),
            f'{"Lower Bounds":15s}{"Upper Bounds":15s}',
            *[f"{l:15.6f}{u:15.6f}" for l, u in zip(self.lb, self.ub)],
            new_lines(2)
        ])

        # results
        rows.extend([f'{"Results":84s}', divider()])

        if len(self.violations) > 0:
            rows.extend([
                'No solution found. List of constraints violated below: ',
                *[f"{i + 1:3d}: {n}" for i, n in enumerate(self.violations)]
            ])
        else:
            sln = ''.join(x for x in self.solution) if isinstance(self.solution, abc.Iterable) else self.solution
            rows.extend([
                'Program found a solution',
                f"Solution: [{sln}]",
                new_lines()
            ])

            if (len(self.tight_hin)) == 0:
                rows.append('None of the constraints were tight')
            else:
                rows.extend([
                    'The following inequality constraints were tight: ',
                    *[f'{i + 1:3d}: {n}' for i, n in enumerate(self.tight_hin)]
                ])

        return '\n'.join(rows)

    def __str__(self):
        return self.as_text()


def _create_matrix_constraint(a, b):
    def fn(w):
        return a @ w - b

    return fn


def _create_gradient_func(fn, eps):
    def f(w, grad):
        diag = np.eye(len(w)) * eps
        if grad.size > 0:
            for i, c in enumerate(diag):
                grad[i] = (fn(w + c) - fn(w - c)) / (2 * eps)
        return fn(w)

    return f


def _validate_matrix_constraints(A, b):
    A = np.asarray(A)
    b = np.asarray(b)

    assert A.ndim == 2, '(In)-Equality matrix `A` must be 2 dimensional!'

    if b.size == 1:
        b = np.repeat(float(b), len(A))
    assert b.ndim == 1, '`b` vector must be 1 dimensional or a scalar'

    return A, b
