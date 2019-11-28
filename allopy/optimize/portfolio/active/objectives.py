from allopy import OptData, get_option


class ObjectiveBuilder:
    def __init__(self, data: OptData, cvar_data: OptData, rebalance: bool):
        self.data = data
        self.cvar_data = cvar_data
        self.rebalance = rebalance

    def max_cvar(self, active_cvar: bool):
        """Maximizes the CVaR. This means that we're minimizing the losses"""

        def _obj_max_cvar(w):
            w = self._format_weights(w, active_cvar)
            return self.cvar_data.cvar(w, self.rebalance) * get_option("F.SCALE")

        return _obj_max_cvar

    @property
    def max_returns(self):
        """Objective function to maximize the returns"""

        def _obj_max_returns(w):
            return self.data.expected_return(w, self.rebalance) * get_option("F.SCALE")

        return _obj_max_returns

    @property
    def max_info_ratio(self):
        def _obj_max_sharpe_ratio(w):
            return self.data.sharpe_ratio([0, *w[1:]], self.rebalance) * get_option("F.SCALE")

        return _obj_max_sharpe_ratio

    def min_volatility(self, is_tracking_error: bool):
        def _obj_min_tracking_error(w):
            w = self._format_weights(w, remove_first_value=is_tracking_error)
            return self.data.volatility(w) * get_option("F.SCALE")

        return _obj_min_tracking_error

    @staticmethod
    def _format_weights(w, remove_first_value: bool):
        return [0, *w[1:]] if remove_first_value else w
