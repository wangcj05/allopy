import numpy as np
from functools import lru_cache, wraps


def lru_cache_ext(maxsize=128, typed=False):
    """
    Extended lru cache that can cache mutable objects like numpy arrays. Does not work with dictionary arguments.

    Examples
    --------
    >>> from allopy.utility import lru_cache_ext
    >>> import time
    >>> @lru_cache_ext()
    ... def calc(data: np.ndarray):
    ...     time.sleep(5)
    ...     return data.sum()

    >>> data = np.random.standard_exponential((10, 10, 10))
    >>> t = time.time()
    >>> calc(data)
    >>> print(round(time.time() - t, 5))

    # repeat the same calculation
    >>> t = time.time()
    >>> calc(data)
    >>> print(round(time.time() - t, 5))
    """

    def decorator(func):
        # Internal store that is used to pass data between the function called by the user and the calculator function
        # that the lru_cache works on.
        store = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            fn_hash = ""
            for a in args:
                fn_hash += _hash(a)

            for k, v in kwargs.items():
                fn_hash += _hash(v, k)
            store[fn_hash] = (args, kwargs)  # save all the arguments to the store
            return cached_caller(fn_hash)

        @lru_cache(maxsize, typed)
        def cached_caller(h):
            args, kwargs = store.pop(h)  # get the arguments data from store
            return func(*args, **kwargs)

        def _hash(data, k='') -> str:
            """This is used to calculate a pseudo-unique hash for the ndarray"""
            try:
                _h = str(hash(data))
            except TypeError:
                data = np.asarray(data)
                _h = str(hash(str(data)))

            return _h + k

        wrapper.cache_info = cached_caller.cache_info
        wrapper.cache_clear = cached_caller.cache_clear

        return wrapper

    return decorator
