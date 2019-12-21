from typing import Iterable, Union

import numpy as np
from scipy.stats import chi2

from .abstract import Penalty


class UncertaintyPenalty(Penalty):
    def __init__(self, uncertainty: Union[Iterable[Union[int, float]], np.ndarray],
                 alpha: float = 0.95,
                 method='direct'):
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

        If using :math:`\chi^2` method, the :math:`\lambda` value is given by

        .. math::
            \lambda = \frac{1}{\chi^2_{n - 1}(\alpha)}

        where :math:`n` is the number of asset classes and :math:`\alpha` is the confidence interval. Otherwise
        the "direct" method will have :math:`\lambda = \alpha`.

        Parameters
        ----------
        uncertainty:
            A 1D vector or 2D matrix representing the uncertainty for the given asset class. If a 1D vector is
            provided, it will be converted to a diagonal matrix

        alpha:
            A constant controlling the intensity of the penalty

        method: "chi2" or "direct"
            Method used to construct the lambda parameter. If "direct", the exact value specified by the `alpha`
            parameter is used. If "chi2", the value is determined using the inverse of the chi-square quantile
            function. In that instance, the `alpha` parameter will be the confidence level. See Notes.
        """
        self._uncertainty = self._derive_uncertainty(np.asarray(uncertainty))
        self.dim = len(self._uncertainty)
        self._method = method.lower()
        self._alpha = self._derive_lambda(alpha, self._method, self.dim)

    def cost(self, w: np.ndarray) -> float:
        r"""
        Calculates the penalty to apply

        .. math::
            p(w) = \lambda \sqrt{w^T \Phi w}
        """
        return self._alpha * (w @ self._uncertainty @ w) ** 0.5

    @property
    def uncertainty(self):
        return self._uncertainty

    @staticmethod
    def _derive_lambda(value: float, method: str, dim: int):
        assert method in ('chi2', 'direct'), f"Unknown method: {method}. Use 'chi2' or 'direct'"
        if method == "direct":
            return value
        else:
            assert 0 < value <= 1, "lambda_ (alpha) parameter must be between 0 and 1 (inclusive) if using 'chi2'"
            return 1 / chi2.ppf(value, dim - 1)

    @staticmethod
    def _derive_uncertainty(uncertainty: np.ndarray):
        if uncertainty.ndim == 1:
            uncertainty = np.diag(uncertainty)

        assert uncertainty.ndim == 2, "uncertainty input must be 1 or 2 dimensional"
        return uncertainty

    def __str__(self):
        arr = repr(self._uncertainty.round(4)).replace("array(", "").replace(")", "")

        return f"""
UncertaintyPenalty(
    lambda={self._alpha},
    uncertainty={arr},
    method={self._method}
)        
        """.strip()
