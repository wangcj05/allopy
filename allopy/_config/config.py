_global_config = {
    'C.SCALE': 1e3,  # constraint function multiplier
    'EPS.CONSTRAINT': 1e-7,
    'EPS.F_ABS': 1e-6,
    'EPS.F_REL': 1e-6,
    'EPS.X_ABS': 1e-6,
    'EPS.X_REL': 0,
    'EPS.STEP': 5e-7,
    'F.SCALE': 1e3,  # objective function multiplier
    'MAX.EVAL': -1,
}


def get_option(key):
    """
    Retrieves the value of the specified option.

    Parameters
    ----------
    key: str
        Name of the option

    Returns
    -------
    result: float or int
        Value of the option

    Raises
    ------
    KeyError:
        Configuration specified by the key does not exist

    Examples
    --------
    >>> from allopy import get_option
    >>> get_option('EPS.CONSTRAINT')  # ok
    >>> try:
    ...     get_option('KEY_DOES_NOT_EXIST')
    ... except KeyError:
    ...     pass  # raises key error
    """
    result = _global_config.get(key.upper(), None)
    if result is None:
        raise KeyError(f"Configuration {key} does not exist.")
    return result


def set_option(key, value):
    """
    Sets the value of the specified option

    Parameters
    ----------
    key: str
        Name of the option

    value: float, int
        Value to set option as

    Notes
    -----
    The available options with its descriptions

    EPS.CONSTRAINT
        The maximum value for which a breach would mean that the constraint is violated

    EPS.FUNCTION
        The maximum value to identify significant changes between 2 (objective) function
        values

    EPS.X_ABS
        The maximum value to identify significant changes between 2 solutions

    EPS.STEP
        The step size used for the numerical gradient

    F.SCALE
        Numeric scale applied to function objective and constraint values to increase
        stability. Only applicable for PortfolioOptimizer.

    MAX.EVAL
        The maximum number of evaluations before the optimizer terminates

    Examples
    --------
    >>> from allopy import set_option
    >>> set_option('EPS.FUNCTION', 1e-6)  # ok
    >>> try:
    ...     set_option('KEY_DOES_NOT_EXIST', 123)
    ... except KeyError:
    ...     pass  # raises key error
    """

    if key.upper() not in _global_config:
        raise KeyError(f"Configuration {key} does not exist.")

    _global_config.update(_validated_key_value(key, value))


def _validated_key_value(key: str, value):
    if key.upper() not in _global_config:
        raise KeyError(f"Configuration {key} does not exist.")

    key = key.upper()
    if key in {'EPS.CONSTRAINT', 'EPS.F_ABS', 'EPS.F_REL', 'EPS.X_ABS', 'EPS.X_REL', 'EPS.STEP', 'F.SCALE'}:
        assert isinstance(value, (int, float)) and value >= 0, "value must be positive number"
    elif key in {'MAX.EVAL'}:
        assert isinstance(value, int), "value must be an integer"

    return {key: value}
