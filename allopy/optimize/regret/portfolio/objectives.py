from allopy import OptData, get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, percentile: float):
        """Maximizes the CVaR. This means that we're minimizing the losses"""

        def objective_creator(d: OptData, i: int):
            def objective(w):
                return (d.cvar(w, self.rebalance, percentile) - self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.cvar_data)]

    @property
    def max_returns(self):
        """Objective function to maximize the returns"""

        def objective_creator(d: OptData, i: int):
            def objective(w):
                return (d.expected_return(w, self.rebalance) - self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.data)]

    @property
    def max_sharpe_ratio(self):
        def objective_creator(d: OptData, i: int):
            def objective(w):
                return (d.sharpe_ratio(w, self.rebalance) - self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.data)]

    @property
    def min_vol(self):
        def objective_creator(d: OptData, i: int):
            def objective(w):
                return (d.volatility(w) + self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.data)]
