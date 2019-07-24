from typing import Callable, Dict, List

import numpy as np

ConstraintMap = Dict[str, Callable[[np.ndarray, np.ndarray], float]]
ConstraintFuncMap = Dict[str, List[Callable[[np.ndarray, np.ndarray], float]]]


class Result:
    pass
