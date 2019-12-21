Portfolio Regret Optimizer
==========================

The :class:`PortfolioRegretOptimizer` inherits the :class:`RegretOptimizer`. The `minimum regret optimization <https://en.wikipedia.org/wiki/Regret_(decision_theory)>`_ is a technique under decision theory on making decisions under uncertainty.

The methods in the :class:`PortfolioRegretOptimizer` are only applied at the first stage of the procedure. The :class:`PortfolioRegretOptimizer` houses the following convenience methods:

:maximize_returns:
    Maximize the returns of the portfolio. You may put in volatility or CVaR constraints for this procedure.

:minimize_volatility:
    Minimizes the total portfolio volatility

:minimize_cvar:
    Minimizes the conditional value at risk (expected shortfall of the portfolio)

:maximize_sharpe_ratio:
    Maximizes the Sharpe ratio of the portfolio.

.. autoclass:: allopy.optimize.PortfolioRegretOptimizer
    :members:
