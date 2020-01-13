from typing import Iterable, Union

from allopy import OptData, get_option
from ..abstract import AbstractConstraintBuilder


class ConstraintBuilder(AbstractConstraintBuilder):
    def max_vol(self, max_vol: Union[Iterable[float], float], as_tracking_error: bool):
        """Volatility must be less than max_vol"""
        max_vol = self._format_constraint(max_vol, "Max volatility")

        def constraint_creator(d: OptData, vol: float):
            def constraint(w):
                w = format_weights(w, as_tracking_error)
                return get_option("C.SCALE") * (d.volatility(w) - vol)

            return constraint

        return [constraint_creator(d, v) for d, v in zip(self.data, max_vol)]

    def max_cvar(self, max_cvar: Union[Iterable[float], float], percentile: float, as_active_cvar: bool):
        """CVaR must be greater than max_cvar"""
        max_cvar = self._format_constraint(max_cvar, "Max CVaR")

        def constraint_creator(d: OptData, cvar: float):
            def constraint(w):
                w = format_weights(w, as_active_cvar)
                return get_option("C.SCALE") * (cvar - d.cvar(w, self.rebalance, percentile))

            return constraint

        return [constraint_creator(d, v) for d, v in zip(self.cvar_data, max_cvar)]

    def min_returns(self, min_ret: Union[Iterable[float], float], as_active_returns):
        """Minimum returns constraint. This is used when objective is to minimize risk st to some minimum returns"""
        min_ret = self._format_constraint(min_ret, "Min returns")

        def constraint_creator(d: OptData, ret: float):
            def constraint(w):
                w = format_weights(w, as_active_returns)
                return get_option("C.SCALE") * (ret - d.expected_return(w, self.rebalance))

            return constraint

        return [constraint_creator(d, v) for d, v in zip(self.data, min_ret)]

    def _format_constraint(self, constraint: Union[Iterable[float], float], name: str):
        if isinstance(constraint, (int, float)):
            constraint = [float(constraint)] * self.num_scenarios
        else:
            constraint = list(constraint)

        assert len(constraint) == self.num_scenarios, f"{name} constraint  must either be a scalar or a vector with " \
                                                      "the same number of elements as the number of scenarios"

        return constraint


def format_weights(w, active: bool):
    return [0, *w[1:]] if active else w
