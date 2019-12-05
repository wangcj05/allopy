from abc import ABC
from typing import Optional, Union
from typing import TypeVar

import numpy as np
from nlopt import RoundoffLimited

from allopy import OptData
from allopy.penalty import Penalty
from allopy.types import OptArray
from ..algorithms import LD_SLSQP
from ..base import BaseOptimizer

__all__ = ['AbstractPortfolioOptimizer', 'AbstractObjectiveBuilder', 'AbstractConstraintBuilder']

PenaltyClass = TypeVar("PenaltyClass", bound=Penalty)


class AbstractPortfolioOptimizer(BaseOptimizer, ABC):
    def __init__(self,
                 data: Union[np.ndarray, OptData],
                 algorithm=LD_SLSQP,
                 cvar_data: Optional[Union[np.ndarray, OptData]] = None,
                 rebalance=False,
                 time_unit='quarterly',
                 sum_to_1=True,
                 *args,
                 **kwargs):
        """
        The PortfolioOptimizer houses several common pre-specified optimization regimes.

        The portfolio optimizer is ideal for modelling under certainty. That is, the portfolio is expected to
        undergo a single fixed scenario in the future.

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

        sum_to_1: bool
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
        if not isinstance(data, OptData):
            data = OptData(data, time_unit)

        if cvar_data is None:
            cvar_data = data.copy().cut_by_horizon(3)
        elif isinstance(cvar_data, (np.ndarray, list, tuple)):
            cvar_data = np.asarray(cvar_data)
            assert cvar_data.ndim == 3, "Must pass in a 3D array for cvar data"
            cvar_data = OptData(cvar_data, time_unit)

        assert isinstance(data, OptData), "data must be an OptData instance"
        assert isinstance(cvar_data, OptData), "cvar_data must be an OptData instance"

        super().__init__(data.n_assets, algorithm, *args, **kwargs)
        self.data: OptData = data
        self.cvar_data: OptData = cvar_data

        self._rebalance = rebalance
        self._max_attempts = kwargs.get('max_attempts', 100)

        self._objectives = None
        self._constraints = None

        if sum_to_1:
            self.add_equality_constraint(lambda w: sum(w) - 1)

    @property
    def max_attempts(self):
        return self._max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int):
        assert isinstance(value, int) and value > 0, 'max_attempts must be an integer >= 1'
        self._max_attempts = value

    @property
    def rebalance(self):
        return self._rebalance

    @rebalance.setter
    def rebalance(self, rebal: bool):
        assert isinstance(rebal, bool), 'rebalance parameter must be boolean'
        self._rebalance = rebal

    @property
    def penalty(self):
        return self._objectives.penalty_class

    @penalty.setter
    def penalty(self, penalty: Optional[PenaltyClass]):
        self._objectives.penalty_class = penalty

    @penalty.deleter
    def penalty(self):
        del self._objectives.penalty_class

    def optimize(self,
                 x0: OptArray = None,
                 *args,
                 initial_solution: Optional[str] = "random",
                 random_state: Optional[int] = None) -> np.ndarray:
        for _ in range(self.max_attempts):
            try:
                w = super().optimize(
                    x0,
                    *args,
                    initial_solution=initial_solution,
                    random_state=random_state
                )

                if w is not None and not np.isnan(w).any():
                    return w

            except (RoundoffLimited, RuntimeError):
                if x0 == "random":
                    x0 = np.random.uniform(self.lower_bounds, self.upper_bounds)
                else:
                    initial_solution = "min_constraint_norm"
        else:
            if self._verbose:
                print('No solution was found for the given problem. Check the summary() for more information')
            return np.repeat(np.nan, self.data.n_assets)


class AbstractObjectiveBuilder(ABC):
    def __init__(self, data: OptData, cvar_data: OptData, rebalance: bool):
        self.data = data
        self.cvar_data = cvar_data
        self.rebalance = rebalance
        self._penalty: Optional[PenaltyClass] = None

    def penalty(self, w: np.ndarray):
        if self._penalty is None:
            return 0.0
        return self._penalty.cost(w)

    @property
    def penalty_class(self):
        return self._penalty

    @penalty_class.setter
    def penalty_class(self, penalty):
        assert isinstance(penalty, Penalty) or None, "value must subclass the Penalty class or be None"
        if penalty is not None:
            assert penalty.dim == self.data.n_assets, "dimension of the penalty does not match the data"

        self._penalty = penalty

    @penalty_class.deleter
    def penalty_class(self):
        self._penalty = None


class AbstractConstraintBuilder(ABC):
    def __init__(self, data: OptData, cvar_data: OptData, rebalance: bool):
        self.data = data
        self.cvar_data = cvar_data
        self.rebalance = rebalance
