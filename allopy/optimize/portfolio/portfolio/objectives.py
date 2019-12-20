from allopy import get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, percentile: float):
        """Maximizes the CVaR. This means that we're minimizing the losses"""

        def _obj_max_cvar(w):
            fv = self.cvar_data.cvar(w, self.rebalance, percentile)
            return (fv - self.penalty(w)) * get_option("F.SCALE")

        return _obj_max_cvar

    @property
    def max_returns(self):
        """Objective function to maximize the returns"""

        def _obj_max_returns(w):
            fv = self.data.expected_return(w, self.rebalance)
            return (fv - self.penalty(w)) * get_option("F.SCALE")

        return _obj_max_returns

    @property
    def max_sharpe_ratio(self):
        def _obj_max_sharpe_ratio(w):
            fv = self.data.sharpe_ratio(w, self.rebalance)
            return (fv - self.penalty(w)) * get_option("F.SCALE")

        return _obj_max_sharpe_ratio

    @property
    def min_vol(self):
        def _obj_min_vol(w):
            fv = self.data.volatility(w)
            return (fv + self.penalty(w)) * get_option("F.SCALE")

        return _obj_min_vol
