from typing import Optional, Union

from copulae.types import Array, Numeric, OptNumeric

__all__ = [
    "Array",
    "Numeric",
    "Real",
    "OptArray",
    "OptNumeric",
    "OptReal",
]

Real = Union[int, float]

OptArray = Optional[Array]
OptReal = Optional[Real]
