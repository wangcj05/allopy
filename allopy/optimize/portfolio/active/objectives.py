from allopy import get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, active_cvar: bool, percentile: float):
        def objective(w):
            w = self._format_weights(w, active_cvar)
            return (self.cvar_data.cvar(w, self.rebalance, percentile) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def max_returns(self):
        def objective(w):
            return (self.data.expected_return(w, self.rebalance) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def max_sharpe_ratio(self):
        def objective(w):
            return (self.data.sharpe_ratio(w, self.rebalance) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    @property
    def max_info_ratio(self):
        def objective(w):
            w = self._format_weights(w, True)
            return (self.data.sharpe_ratio(w, self.rebalance) - self.penalty(w)) * get_option("F.SCALE")

        return objective

    def min_volatility(self, is_tracking_error: bool):
        def objective(w):
            w = self._format_weights(w, remove_first_value=is_tracking_error)
            return (self.data.volatility(w) + self.penalty(w)) * get_option("F.SCALE")

        return objective

    @staticmethod
    def _format_weights(w, remove_first_value: bool):
        return [0, *w[1:]] if remove_first_value else w
