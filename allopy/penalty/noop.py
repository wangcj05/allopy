import numpy as np

from .abstract import Penalty


class NoPenalty(Penalty):
    def __init__(self, dim: int):
        self.dim = dim

    def cost(self, _: np.ndarray) -> float:
        return 0

    def __str__(self):
        return f"NoPenalty(dim={self.dim})"
