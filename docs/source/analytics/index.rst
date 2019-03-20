Analytics
=========

.. module:: allopy.analytics
.. py:currentmodule:: allopy.analytics

.. automodule:: allopy.analytics

The analytics subpackage contains a repository of functions to gauge the portfolio given a particular weights. It is useful to run functions in this subpackage after deriving a set of optimal weights with the optimizers.

Attribution
~~~~~~~~~~~

.. autofunction:: cvar_attr
.. autofunction:: returns_attr
.. autofunction:: risk_attr

Beta Correlation
~~~~~~~~~~~~~~~~

.. autofunction:: beta_corr

Drawdowns
~~~~~~~~~

.. autofunction:: drawdowns
.. autofunction:: drawdown_stats

Metrics
~~~~~~~

.. autofunction:: performance_ratios
.. autofunction:: performance_ratios_
.. autofunction:: returns_metrics
.. autofunction:: risk_metrics

Returns Path
~~~~~~~~~~~~

.. autofunction:: cumulative_returns_path

Sensitivity
~~~~~~~~~~~

.. autofunction:: sensitivity_cvar
.. autofunction:: sensitivity_returns
.. autofunction:: sensitivity_vol
