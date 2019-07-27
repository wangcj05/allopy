import pytest
from numpy.testing import assert_almost_equal

from allopy import BaseOptimizer


@pytest.fixture("module")
def model():
    model = BaseOptimizer(2)

    def obj(w):
        return 100 * (w[1] - w[0] ** 2) ** 2 - (1 - w[0]) ** 2

    def c1(w):
        return w[0] ** 2 + w[1] - 1

    def c2(w):
        return w[0] ** 2 - w[1] - 1

    model.set_min_objective(obj)
    model.add_inequality_constraint(c1)
    model.add_inequality_constraint(c2)
    model.add_inequality_matrix_constraint([[1, 2]], [1])
    model.add_equality_matrix_constraint([[2, 1]], [1])
    model.set_bounds([0, -0.5], [1, 2])
    model.optimize([0.5, 0.5])

    return model


@pytest.mark.parametrize("initial_solution, random_state", [
    ("random", 888),
    ("min_constraint_norm", 888),
])
def test_constrained_optimization(initial_solution, random_state):
    r"""
    Maximize the Rosenbrock function

    .. math::
        \max 100(y−x^2)^2 + (1 − x)^2 \\
        s.t. \\
        x + 2y \leq 1 \\
        x^2 + y \leq 1 \\
        x^2 - y \leq 1 \\
        2x + y = 1 \\
        0 \leq x \leq 1 \\
        0.5 \leq y \leq 2
    """
    model = BaseOptimizer(2)

    def obj(w):
        return 100 * (w[1] - w[0] ** 2) ** 2 - (1 - w[0]) ** 2

    def c1(w):
        return w[0] + 2 * w[1] - 1

    def c2(w):
        return w[0] ** 2 + w[1] - 1

    def c3(w):
        return w[0] ** 2 - w[1] - 1

    def c4(w):
        return 2 * w[0] + w[1] - 1

    model.set_min_objective(obj)
    model.add_inequality_constraint(c1)
    model.add_inequality_constraint(c2)
    model.add_inequality_constraint(c3)
    model.add_equality_constraint(c4)
    model.set_bounds([0, -0.5], [1, 2])
    w = model.optimize(initial_solution=initial_solution, random_state=random_state)

    expected = [0.41494475, 0.1701105]
    assert obj(w) < obj(expected) or assert_almost_equal(w, expected)


def test_matrix_constrained_optimization():
    r"""
    Maximize the Rosenbrock function

    .. math::
        \max 100(y−x^2)^2 + (1 − x)^2 \\
        s.t. \\
        x + 2y \leq 1 \\
        x^2 + y \leq 1 \\
        x^2 - y \leq 1 \\
        2x + y = 1 \\
        0 \leq x \leq 1 \\
        0.5 \leq y \leq 2
    """
    model = BaseOptimizer(2)

    def obj(w):
        return 100 * (w[1] - w[0] ** 2) ** 2 - (1 - w[0]) ** 2

    def c1(w):
        return w[0] ** 2 + w[1] - 1

    def c2(w):
        return w[0] ** 2 - w[1] - 1

    model.set_min_objective(obj)
    model.add_inequality_constraint(c1)
    model.add_inequality_constraint(c2)
    model.add_inequality_matrix_constraint([[1, 2]], [1])
    model.add_equality_matrix_constraint([[2, 1]], [1])
    model.set_bounds([0, -0.5], [1, 2])
    w = model.optimize([0.5, 0.5])

    expected = [0.41494475, 0.1701105]
    assert obj(w) < obj(expected) or assert_almost_equal(w, expected)


def test_summary(model):
    assert model.summary() is not None
