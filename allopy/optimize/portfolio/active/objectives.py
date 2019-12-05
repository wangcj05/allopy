from allopy import get_option
from ..abstract import AbstractObjectiveBuilder


class ObjectiveBuilder(AbstractObjectiveBuilder):
    def max_cvar(self, active_cvar: bool):
        def _obj_max_cvar(w):
            w = self._format_weights(w, active_cvar)
            fv = self.cvar_data.cvar(w, self.rebalance)
            return (fv - self.penalty(w)) * get_option("F.SCALE")

        return _obj_max_cvar

    @property
    def max_returns(self):
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
    def max_info_ratio(self):
        def _obj_max_info_ratio(w):
            w = self._format_weights(w, True)
            fv = self.data.sharpe_ratio(w, self.rebalance)
            return (fv - self.penalty(w)) * get_option("F.SCALE")

        return _obj_max_info_ratio

    def min_volatility(self, is_tracking_error: bool):
        def _obj_min_tracking_error(w):
            w = self._format_weights(w, remove_first_value=is_tracking_error)
            fv = self.data.volatility(w)
            return (fv + self.penalty(w)) * get_option("F.SCALE")

        return _obj_min_tracking_error

    @staticmethod
    def _format_weights(w, remove_first_value: bool):
        return [0, *w[1:]] if remove_first_value else w
