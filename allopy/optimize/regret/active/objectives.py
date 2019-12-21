from allopy import OptData, get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, percentile: float, active_cvar: bool):
        """Maximizes the CVaR. This means that we're minimizing the losses"""

        def objective_creator(d: OptData, i: int):
            def objective(w):
                w = format_weights(w, active_cvar)
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
    def max_info_ratio(self):
        def objective_creator(d: OptData, i: int):
            def objective(w):
                w = format_weights(w, True)
                return (d.sharpe_ratio(w, self.rebalance) - self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.data)]

    def min_vol(self, is_tracking_error: bool):
        def objective_creator(d: OptData, i: int):
            def objective(w):
                w = format_weights(w, is_tracking_error)
                return (d.volatility(w) + self.penalties[i](w)) * get_option("F.SCALE")

            return objective

        return [objective_creator(d, i) for i, d in enumerate(self.data)]


def format_weights(w, remove_first_value: bool):
    return [0, *w[1:]] if remove_first_value else w
