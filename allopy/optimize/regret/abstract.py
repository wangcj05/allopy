from abc import ABC
from typing import List, Optional, Union

import numpy as np

from allopy import OptData
from allopy.penalty import NoPenalty, Penalty

__all__ = ["AbstractObjectiveBuilder", "AbstractConstraintBuilder"]


class AbstractObjectiveBuilder(ABC):
    def __init__(self, data: List[OptData], cvar_data: List[OptData], rebalance: bool, time_unit):
        self.data, self.cvar_data = format_inputs(data, cvar_data, time_unit)
        self.rebalance = rebalance
        self.num_scenarios = len(data)

        assert self.num_scenarios > 0, "Provide data to the optimizer"
        assert self.num_scenarios == len(cvar_data), "data and cvar data must have same number of scenarios"

        self.num_assets = data[0].n_assets
        assert all(d.n_assets == self.num_assets for d in data), \
            f"number of assets in data should equal {self.num_assets}"

        assert all(d.n_assets == self.num_assets for d in cvar_data), \
            f"number of assets in cvar data should equal {self.num_assets}"

        self._penalties = [NoPenalty(self.num_assets)] * self.num_scenarios

    @property
    def penalties(self):
        return self._penalties

    @penalties.setter
    def penalties(self, penalties):
        assert penalties is None or isinstance(penalties, Penalty) or hasattr(penalties, "__iter__"), \
            "penalties can be None, a subsclass of the Penalty class or a list which subclasses the Penalty class"

        if penalties is None:
            self._penalties = [NoPenalty(self.num_assets)] * self.num_scenarios
        elif isinstance(penalties, penalties):
            self._penalties = [penalties] * self.num_scenarios
        else:
            penalties = list(penalties)
            assert len(penalties) == self.num_scenarios, "number of penalties given must match number of scenarios"
            assert all(isinstance(p, Penalty) for p in penalties), "non-Penalty instance detected"
            self._penalties = penalties


class AbstractConstraintBuilder(ABC):
    def __init__(self, data: List[OptData], cvar_data: List[OptData], rebalance: bool, time_unit):
        self.data, self.cvar_data = format_inputs(data, cvar_data, time_unit)
        self.rebalance = rebalance
        self.num_scenarios = len(self.data)


def format_inputs(data: List[Union[OptData, np.ndarray]],
                  cvar_data: Optional[List[Union[OptData, np.ndarray]]],
                  time_unit: int):
    data = [d if isinstance(data, OptData) else OptData(d, time_unit) for d in data]

    if cvar_data is None:
        return [d.cut_by_horizon(3) for d in data]
    else:
        cvar_data = [c if isinstance(c, OptData) else OptData(c, time_unit) for c in cvar_data]

    return data, cvar_data
