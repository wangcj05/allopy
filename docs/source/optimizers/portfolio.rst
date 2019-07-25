Portfolio Optimizer
===================

The :class:`PortfolioOptimizer` inherits the :class:`BaseOptimizer` to add several convenience methods. These methods include common optimization programs which would be tedious to craft with the :class:`BaseOptimizer` over and over again. Of course, as an extension, it can do anything that the :class:`BaseOptimizer` can. However, if that's the goal, it would be better to stick with the :class:`BaseOptimizer` to reduce confusion when reading the code.

Using the :class:`PortfolioOptimizer` assumes that there is a returns stream from which all other asset classes are benchmarked against. This is the first index in the assets axis.

For example, if you have a benchmark (beta) returns stream, 9 other asset classes over 10000 trials and 40 periods, the simulation tensor will be 40 x 10000 x 10 with the first asset axis being the returns of the benchmark. In such a case, the active portfolio optimizer can be used to optimize the portfolio relative to the benchmark.

.. autoclass:: allopy.optimize.PortfolioOptimizer
    :members:
    :inherited-members:
