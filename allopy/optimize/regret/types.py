from typing import Callable, List, Union

import numpy as np

__all__ = ["Arg1Func", "Arg2Func", "ObjFunc"]

Arg1Func = Callable[[np.ndarray], float]
Arg2Func = Callable[[np.ndarray, np.ndarray], float]
ConFunc = Union[Arg1Func, Arg2Func, List[Arg1Func], List[Arg2Func]]
ObjFunc = Union[Arg1Func, Arg2Func, List[Arg1Func], List[Arg2Func]]
