from allopy import get_option
from ..abstract import AbstractConstraintBuilder


class ConstraintBuilder(AbstractConstraintBuilder):
    def max_vol(self, max_vol: float, as_tracking_error=False):
        """Volatility must be less than max_vol"""

        def _ctr_max_vol(w):
            w = self._active_weights(w, as_tracking_error)
            return get_option("C.SCALE") * (self.data.volatility(w) - max_vol)

        return _ctr_max_vol

    def max_cvar(self, max_cvar: float, percentile=5.0, as_active_cvar=False):
        """CVaR must be greater than max_cvar"""

        def _ctr_max_cvar(w):
            w = self._active_weights(w, as_active_cvar)
            return get_option("C.SCALE") * (max_cvar - self.cvar_data.cvar(w, self.rebalance, percentile))

        return _ctr_max_cvar

    def min_returns(self, min_ret: float, as_active_returns=False):
        """Minimum returns constraint. This is used when objective is to minimize risk st to some minimum returns"""

        def _ctr_min_returns(w):
            w = self._active_weights(w, as_active_returns)
            return get_option("C.SCALE") * (min_ret - self.data.expected_return(w, self.rebalance))

        return _ctr_min_returns

    @staticmethod
    def _active_weights(w, active: bool):
        return [0, *w[1:]] if active else w
