from allopy import OptData, get_option


class ObjectiveBuilder:
    def __init__(self, data: OptData, cvar_data: OptData, rebalance: bool):
        self.data = data
        self.cvar_data = cvar_data
        self.rebalance = rebalance

    @property
    def max_cvar(self):
        """Maximizes the CVaR. This means that we're minizming the losses"""

        def _obj_max_cvar(w):
            return self.cvar_data.cvar(w, self.rebalance) * get_option("F.SCALE")

        return _obj_max_cvar

    @property
    def max_returns(self):
        """Objective function to maximize the returns"""

        def _obj_max_returns(w):
            return self.data.expected_return(w, self.rebalance) * get_option("F.SCALE")

        return _obj_max_returns

    @property
    def max_sharpe_ratio(self):
        def _obj_max_sharpe_ratio(w):
            return self.data.sharpe_ratio(w, self.rebalance) * get_option("F.SCALE")

        return _obj_max_sharpe_ratio

    @property
    def min_vol(self):
        def _obj_min_vol(w):
            return self.data.volatility(w) * get_option("F.SCALE")

        return _obj_min_vol
