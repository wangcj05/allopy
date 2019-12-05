from allopy import get_option
from ..abstract import AbstractConstraintBuilder


class ConstraintBuilder(AbstractConstraintBuilder):

    def max_vol(self, max_vol: float):
        """Volatility must be less than max_vol"""

        def _ctr_max_vol(w):
            return get_option("F.SCALE") * (self.data.volatility(w) - max_vol)

        return _ctr_max_vol

    def max_cvar(self, max_cvar: float, percentile=5.0):
        """CVaR must be greater than max_cvar"""

        def _ctr_max_cvar(w):
            return get_option("F.SCALE") * (max_cvar - self.cvar_data.cvar(w, self.rebalance, percentile))

        return _ctr_max_cvar

    def min_returns(self, min_ret: float):
        """Minimim returns constraint. This is used when objective is to minimize risk st to some minimum returns"""

        def _ctr_min_returns(w):
            return get_option("F.SCALE") * (min_ret - self.data.expected_return(w, self.rebalance))

        return _ctr_min_returns
