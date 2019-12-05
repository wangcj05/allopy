API
===

The optimizers are the classes responsible for finding the ideal weights. The underlying optimizer is built from the `nlopt <https://nlopt.readthedocs.io/en/latest/>`_ package. The optimizers are classified into 2 categories, **Deterministic** and **Discrete State Uncertainty**.

**Deterministic optimizers** are suitable for instances where the problem scenario is known. For example, if it is expected to only have one market scenario, then it is suitable to use the optimizers under this category.

**Discrete State Uncertainty optimizers** are suitable for instances where there are multiple problem scenarios and when a discrete probability can be assigned to each scenario. For instance, there could be 3 scenarios that would happen in the future. Baseline at 50%, Upside at 30% and Downside at 20%. In this case, this class of optimizers will be most suitable to model the optimization problem.

Presently, **Continuous State Uncertainty optimizers** are not implemented in the package.

.. toctree::
    :caption: Deterministic Optimizers

    Base Optimizer <base>
    Portfolio Optimizer <portfolio>
    Active Portfolio Optimizer <active-portfolio>

.. toctree::
    :caption: Discrete State Uncertainty Optimizers

    Regret Optimizer <regret>
    Portfolio Regret Optimizer <portfolio-regret>


Algorithms
----------

All algorithms in :code:`allopy` follows a particular naming pattern. Specifically, the names are of the form :code:`{G,L}{N,D}_xxx` where :code:`G`/:code:`L` denotes if the algorithm is **global** or **local** optimization, :code:`N`/:code:`D` denotes if the algorithm is a **derivative-free** or **gradient-based** algorithm and :code:`xxx` is the name of the algorithm.

For example, :code:`LD_SLSQP` refers to the Sequential Quadratic Least Squares Programming. This is a local and derivative free algorithm.

Many algorithms have variants and some, such as :code:`AUGLAG`, may not have the construct described above. In using any of these algorithms, consult a textbook or Wikipedia to understand if it is fit for your purpose.

A list of the algorithms is given below. `Check out the docs at nlopt for more information <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms>`_.

* GN_DIRECT_L
* GN_DIRECT_L_RAND
* GN_DIRECT_NOSCAL
* GN_DIRECT_L_NOSCAL
* GN_DIRECT_L_RAND_NOSCAL
* GN_ORIG_DIRECT
* GN_ORIG_DIRECT_L
* GD_STOGO
* GD_STOGO_RAND
* LD_LBFGS_NOCEDAL
* LD_LBFGS
* LN_PRAXIS
* LD_VAR1
* LD_VAR2
* LD_TNEWTON
* LD_TNEWTON_RESTART
* LD_TNEWTON_PRECOND
* LD_TNEWTON_PRECOND_RESTART
* GN_CRS2_LM
* GN_MLSL
* GD_MLSL
* GN_MLSL_LDS
* GD_MLSL_LDS
* LD_MMA
* LN_COBYLA
* LN_NEWUOA
* LN_NEWUOA_BOUND
* LN_NELDERMEAD
* LN_SBPLX
* LN_AUGLAG
* LD_AUGLAG
* LN_AUGLAG_EQ
* LD_AUGLAG_EQ
* LN_BOBYQA
* GN_ISRES
* AUGLAG
* AUGLAG_EQ
* G_MLSL
* G_MLSL_LDS
* LD_SLSQP
* LD_CCSAQ
* GN_ESCH
* GN_AGS
