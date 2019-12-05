from typing import Iterable, Union

import numpy as np

from .abstract import Penalty


class UncertaintyPenalty(Penalty):
    def __init__(self, uncertainty: Union[Iterable[Union[int, float]], np.ndarray], lambda_=1):
        r"""
        The uncertainty penalty. It penalizes the objective function relative to the level of uncertainty for the
        given asset

        Notes
        -----
        Given an initial maximizing objective, this penalty will change the objective to

        .. math::
            f(w) - \lambda \sqrt{w^T \Phi w}

        where :math:`\Phi` represent the uncertainty matrix. :math:`\lambda = 0` or a 0-matrix is a special case
        where there are no uncertainty in the projections.

        Parameters
        ----------
        uncertainty:
            A 1D vector or 2D matrix representing the uncertainty for the given asset class. If a 1D vector is
            provided, it will be converted to a diagonal matrix

        lambda_:
            A constant controlling the intensity of the penalty
        """

        self._lambda = lambda_
        self._uncertainty = np.asarray(uncertainty)
        if self._uncertainty.ndim == 1:
            self._uncertainty = np.diag(self._uncertainty)

        self.dim = len(self._uncertainty)
        assert self._uncertainty.ndim == 2, ""

    def cost(self, w: np.ndarray) -> float:
        return self._lambda * (w @ self._uncertainty @ w) ** 0.5

    def __str__(self):
        arr = repr(self._uncertainty.round(4)).replace("array(", "").replace(")", "")

        return f"""
UncertaintyPenalty(
    lambda={self._lambda},
    uncertainty={arr}
)        
        """.strip()
