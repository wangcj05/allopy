import numpy as np

from .abstract import Penalty


class NoPenalty(Penalty):
    def __init__(self, dim: int):
        """
        No penalty is a no-op penalty function. Essentially it applies no penalty to the objective function
        when applied to the objective function.

        Parameters
        ----------
        dim
            Number of assets
        """
        self.dim = dim

    def cost(self, _: np.ndarray) -> float:
        r"""
        Calculates the penalty to apply

        .. math::
            p(w) = 0
        """
        return 0

    def __str__(self):
        return f"NoPenalty(dim={self.dim})"
