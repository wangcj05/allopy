from typing import Optional, Union

import numpy as np

from allopy import OptData
from allopy.types import OptArray, OptReal
from .abstract import AbstractPortfolioOptimizer
from .obj_ctr import *
from ..algorithms import LD_SLSQP


class PortfolioOptimizer(AbstractPortfolioOptimizer):
    def __init__(self,
                 data: Union[np.ndarray, OptData],
                 algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None,
                 rebalance=False,
                 time_unit: int = 'quarterly',
                 sum_to_1=True,
                 *args,
                 **kwargs):
        """
        PortfolioOptimizer houses several common pre-specified optimization routines.

        PortfolioOptimizer assumes that the optimization model has no uncertainty. That is, the portfolio is
        expected to undergo a single fixed scenario in the future. By default, the PortfolioOptimizer will
        automatically add an equality constraint that forces the portfolio weights to sum to 1.

        Parameters
        ----------
        data: {ndarray, OptData}
            The data used for optimization

        algorithm: {int, string}
            The algorithm used for optimization. Default is Sequential Least Squares Programming

        cvar_data: {ndarray, OptData}
            The cvar_data data used as constraint during the optimization. If this is not set, will default to being a
            copy of the original data that is trimmed to the first 3 years. If an array like object is passed in,
            the data must be a 3D array with axis representing time, trials and assets respectively. In that
            instance, the horizon will not be cut at 3 years, rather it'll be left to the user.

        rebalance: bool, optional
            Whether the weights are rebalanced in every time instance. Defaults to False

        time_unit: {int, 'monthly', 'quarterly', 'semi-annually', 'yearly'}, optional
            Specifies how many units (first axis) is required to represent a year. For example, if each time period
            represents a month, set this to 12. If quarterly, set to 4. Defaults to 12 which means 1 period represents
            a month. Alternatively, specify one of 'monthly', 'quarterly', 'semi-annually' or 'yearly'

        sum_to_1:
            If True, portfolio weights must sum to 1.

        args:
            other arguments to pass to the :class:`BaseOptimizer`

        kwargs:
            other keyword arguments to pass into :class:`OptData` (if you passed in a numpy array for `data`) or into
            the :class:`BaseOptimizer`

        See Also
        --------
        :class:`BaseOptimizer`: Base Optimizer
        :class:`OptData`: Optimizer data wrapper
        """
        super().__init__(data, algorithm, cvar_data, rebalance, time_unit, sum_to_1, *args, **kwargs)

    def maximize_returns(self,
                         max_vol: OptReal = None,
                         max_cvar: OptReal = None,
                         x0: OptArray = None,
                         tol=0.0,
                         initial_solution: Optional[str] = "random",
                         random_state: Optional[int] = None) -> np.ndarray:
        """
        Optimizes the expected returns of the portfolio subject to max volatility and/or cvar constraint.
        At least one of the tracking error or cvar constraint must be defined.

        If `max_vol` is defined, the tracking error will be offset by that amount. Maximum tracking error is usually
        defined by a positive number. Meaning if you would like to cap tracking error to 3%, max_te should be set to
        0.03.

        Parameters
        ----------
        max_vol: scalar, optional
            Maximum tracking error allowed

        max_cvar: scalar, optional
            Maximum cvar_data allowed

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """

        assert not (max_vol is None and max_cvar is None), "If maximizing returns subject to some sort of vol/CVaR " \
                                                           "constraint, we must at least specify max CVaR or max vol"

        if max_vol is not None:
            self.add_inequality_constraint(ctr_max_vol(self.data, max_vol), tol)

        if max_cvar is not None:
            self.add_inequality_constraint(ctr_max_cvar(self.cvar_data, max_cvar, self.rebalance), tol)

        self.set_max_objective(obj_max_returns(self.data, self.rebalance))
        return self.optimize(x0, initial_solution=initial_solution, random_state=random_state)

    def minimize_volatility(self,
                            min_ret: OptReal = None,
                            x0: OptArray = None,
                            tol=0.0,
                            initial_solution: Optional[str] = "random",
                            random_state: Optional[int] = None) -> np.ndarray:
        """
        Minimizes the tracking error of the portfolio

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are
        at least as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(ctr_min_returns(self.data, min_ret, self.rebalance), tol)

        self.set_min_objective(obj_min_vol(self.data))
        return self.optimize(x0, initial_solution=initial_solution, random_state=random_state)

    def minimize_cvar(self,
                      min_ret: OptReal = None,
                      x0: OptArray = None,
                      tol=0.0,
                      initial_solution: Optional[str] = "random",
                      random_state: Optional[int] = None) -> np.ndarray:
        """
        Minimizes the conditional value at risk of the portfolio. The present implementation actually minimizes the
        expected shortfall.

        If the `min_ret` is specified, the optimizer will search for an optimal portfolio where the returns are at least
        as large as the value specified (if possible).

        Parameters
        ----------
        min_ret: float, optional
            The minimum returns required for the portfolio

        x0: ndarray
            Initial vector. Starting position for free variables

        tol: float
            A tolerance for the constraints in judging feasibility for the purposes of stopping the optimization

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        if min_ret is not None:
            self.add_inequality_constraint(ctr_min_returns(self.data, min_ret, self.rebalance), tol)

        self.set_max_objective(obj_max_cvar(self.data, self.rebalance))
        return self.optimize(x0, initial_solution=initial_solution, random_state=random_state)

    def maximize_sharpe_ratio(self,
                              x0: OptArray = None,
                              initial_solution: Optional[str] = "random",
                              random_state: Optional[int] = None) -> np.ndarray:
        """
        Maximizes the sharpe ratio the portfolio.

        Parameters
        ----------
        x0: array_like, optional
            Initial vector. Starting position for free variables

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        initial_solution: str, optional
            The method to find the initial solution if the initial vector :code:`x0` is not specified. Set as
            :code:`None` to disable. However, if disabled, the initial vector must be supplied.

        random_state: int, optional
            Random seed. Applicable if :code:`initial_solution` is not :code:`None`

        Returns
        -------
        ndarray
            Optimal weights
        """
        self.set_max_objective(obj_max_sharpe_ratio(self.data, self.rebalance))
        return self.optimize(x0, initial_solution=initial_solution, random_state=random_state)
