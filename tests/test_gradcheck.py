import numpy as np
import pytest

from adaptive_neural_systems.gradcheck import (
    max_z1_change,
    numerical_gradient,
    relative_error,
    relu_kink_margin,
)


def test_numerical_gradient_of_a_known_function():
    # L = sum(a^2) + 3 * b  ->  dL/da = 2a, dL/db = 3
    params = {"a": np.array([[1.0, -2.0], [0.5, 4.0]]), "b": np.array([7.0])}

    def loss_fn(p):
        return float((p["a"] ** 2).sum() + 3.0 * p["b"].sum())

    grads = numerical_gradient(loss_fn, params)
    np.testing.assert_allclose(grads["a"], 2.0 * params["a"], rtol=1e-8)
    np.testing.assert_allclose(grads["b"], [3.0], rtol=1e-8)


def test_numerical_gradient_restores_params():
    params = {"a": np.array([1.0, 2.0, 3.0])}
    before = params["a"].copy()
    numerical_gradient(lambda p: float((p["a"] ** 3).sum()), params)
    np.testing.assert_array_equal(params["a"], before)


def test_central_difference_error_is_second_order():
    # For L = x^3 at x = 1, the central-difference error is exactly eps^2.
    params = {"x": np.array([1.0])}
    eps = 1e-3
    estimate = numerical_gradient(lambda p: float(p["x"][0] ** 3), params, eps)["x"][0]
    assert estimate - 3.0 == pytest.approx(eps**2, rel=1e-4)


def test_relative_error():
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
    assert relative_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert relative_error(np.array([1.0]), np.array([-1.0])) == pytest.approx(1.0)
    # Same relative mismatch at very different scales gives the same error.
    a, b = np.array([1.0]), np.array([1.01])
    assert relative_error(a, b) == pytest.approx(relative_error(1e-8 * a, 1e-8 * b))


def test_kink_helpers():
    assert relu_kink_margin(np.array([[0.5, -0.2], [3.0, -1.0]])) == pytest.approx(0.2)
    assert max_z1_change(np.array([[0.0, 0.5]]), 1e-5) == pytest.approx(1e-5)
    assert max_z1_change(np.array([[0.0, 4.0]]), 1e-5) == pytest.approx(4e-5)
