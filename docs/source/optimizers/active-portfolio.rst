Active Portfolio Optimizer
==========================

The :class:`ActivePortfolioOptimizer` inherits the :class:`BaseOptimizer` to add several convenience methods. These methods include common optimization programs which would be tedious to craft with the :class:`BaseOptimizer` over and over again. Of course, as an extension, it can do anything that the :class:`BaseOptimizer` can. However, if that's the goal, it would be better to stick with the :class:`BaseOptimizer` to reduce confusion when reading the code.

:class:`ActivePortfolioOptimizer` houses the following convenience methods:

:maximize_eva:
    Maximize the expected value added of the portfolio. The objective function is the same with maximize returns as it just maximizes the total returns. For risk constraints, this method by default will constrain on tracking error and total CVaR. That is the volatility (tracking error) is calculated with the first variable (usually a passive portion) set to 0. Total CVaR has no treatments done to it. You can override these defaults in the method itself.

:minimize_tracking_error:
    Minimizes the tracking error of the portfolio. Tracking error is calculated by setting the first variable to 0 whilst the rest are updated by the optimizer.

:minimize_volatility:
    Minimizes the total portfolio volatility

:minimize_cvar:
    Minimizes the conditional value at risk (expected shortfall of the portfolio)

:maximize_info_ratio:
    Maximizes the information ratio of the portfolio. The information ratio of the portfolio is calculated like the Sharpe ratio. The only difference is the first variable is set to 0.

:maximize_sharpe_ratio:
    Maximizes the Sharpe ratio of the portfolio.


.. autoclass:: allopy.optimize.ActivePortfolioOptimizer
    :members:
