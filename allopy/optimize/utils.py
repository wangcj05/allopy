import numpy as np

__all__ = ["create_gradient_func",
           "create_matrix_constraint",
           "validate_matrix_constraints",
           "validate_tolerance"]


def create_gradient_func(fn, eps):
    def f(w, grad):
        diag = np.eye(len(w)) * eps
        if grad.size > 0:
            for i, c in enumerate(diag):
                grad[i] = (fn(w + c) - fn(w - c)) / (2 * eps)
        return fn(w)

    return f


def create_matrix_constraint(a, b, name: str = None):
    def fn(w):
        return a @ w - b

    if name is not None:
        fn.__name__ = name

    return fn


def validate_matrix_constraints(A, b):
    A = np.asarray(A)
    b = np.asarray(b)

    assert A.ndim == 2, '(In)-Equality matrix `A` must be 2 dimensional!'

    if b.size == 1:
        b = np.repeat(float(b), len(A))
    assert b.ndim == 1, '`b` vector must be 1 dimensional or a scalar'

    return A, b


def validate_tolerance(tol):
    if tol is None:
        return None
    elif isinstance(tol, (float, int)):
        assert tol >= 0, "tolerance must be positive"
        return float(tol)
    elif hasattr(tol, "__iter__"):
        tol = np.asarray(tol, dtype=float)
        assert np.all(tol > 0), "tolerance must all be positive"
        return tol
    else:
        raise ValueError("tolerance must either be a float or a vector of floats")
