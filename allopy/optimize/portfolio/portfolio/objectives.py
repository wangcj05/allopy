from allopy import get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, percentile: float):
        """Maximizes the CVaR. This means that we're minimizing the losses"""

        def objective(w):
            return (self.cvar_data.cvar(w, self.rebalance, percentile) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def max_returns(self):
        """Objective function to maximize the returns"""

        def objective(w):
            return (self.data.expected_return(w, self.rebalance) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def max_sharpe_ratio(self):
        def objective(w):
            return (self.data.sharpe_ratio(w, self.rebalance) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def min_vol(self):
        def objective(w):
            return (self.data.volatility(w) + self.penalty(w)) * get_option("F.SCALE")

        return objective
