Penalty
=======

The penalties are classes that are added to the Portfolio Optimizer family of optimizers (i.e. :class:`PortfolioOptimizer`, :class:`ActivePortfolioOptimizer`) to impose a penalty to the particular asset weight based on the amount of uncertainty present for that asset class.

Uncertainty in this instance does not mean the risk (or variance). Rather, it signifies how uncertain we are of those estimates. For example, it represents how uncertain we are of the returns (mean) and volatility (standard deviation) estimates we have projected for the asset class.

.. toctree::
    :caption: Penalty Classes

    Uncertainty Penalty <uncertainty>
