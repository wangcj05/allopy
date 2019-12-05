from abc import ABC, abstractmethod

import numpy as np


class Penalty(ABC):
    dim = 0

    @abstractmethod
    def cost(self, w: np.ndarray) -> float:
        """The cost incurred by the penalty function"""
        pass
