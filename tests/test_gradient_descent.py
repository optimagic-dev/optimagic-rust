"""Tests for the Python bindings of gradient descent.

The algorithm itself is tested in Rust (crates/optimagic-rust/src/gradient_descent.rs).
These tests focus on the Python layer: argument conversion, result types, defaults and
error handling.

"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from optimagic_rust import GradientDescentResult, minimize_gradient_descent

WEIGHTS = np.array([1.0, 2.0, 3.0])
OPTIMUM = np.array([0.0, 1.0, 2.0])


def quadratic(x):
    return float(WEIGHTS @ (x - OPTIMUM) ** 2)


def quadratic_jac(x):
    return 2 * WEIGHTS * (x - OPTIMUM)


def test_minimize_quadratic():
    res = minimize_gradient_descent(quadratic, quadratic_jac, np.array([5.0, -3, 1]))
    assert isinstance(res, GradientDescentResult)
    assert res.converged
    assert_array_almost_equal(res.x, OPTIMUM)
    assert res.fun == pytest.approx(0, abs=1e-10)


def test_result_types():
    res = minimize_gradient_descent(quadratic, quadratic_jac, np.array([5.0, -3, 1]))
    assert isinstance(res.x, np.ndarray)
    assert res.x.dtype == np.float64
    assert res.x.shape == (3,)
    assert isinstance(res.fun, float)
    assert isinstance(res.n_iterations, int)
    assert isinstance(res.n_fun_evals, int)
    assert isinstance(res.n_jac_evals, int)
    assert isinstance(res.converged, bool)
    assert isinstance(res.message, str)


@pytest.mark.parametrize(
    "x0", [[5, -3, 1], (5.0, -3.0, 1.0), np.array([5, -3, 1], dtype=np.int64)]
)
def test_x0_can_be_any_array_like(x0):
    res = minimize_gradient_descent(quadratic, quadratic_jac, x0)
    assert_array_almost_equal(res.x, OPTIMUM)


def test_jac_can_return_list():
    res = minimize_gradient_descent(
        quadratic, lambda x: quadratic_jac(x).tolist(), np.array([5.0, -3, 1])
    )
    assert_array_almost_equal(res.x, OPTIMUM)


def test_x0_is_not_modified():
    x0 = np.array([5.0, -3, 1])
    minimize_gradient_descent(quadratic, quadratic_jac, x0)
    assert_array_almost_equal(x0, [5.0, -3, 1])


def test_callbacks_receive_float_arrays_that_they_may_modify():
    def fun(x):
        assert isinstance(x, np.ndarray)
        assert x.dtype == np.float64
        value = quadratic(x)
        x[:] = 1e10  # Must not affect the algorithm.
        return value

    res = minimize_gradient_descent(fun, quadratic_jac, np.array([5.0, -3, 1]))
    assert_array_almost_equal(res.x, OPTIMUM)


def test_options_are_used():
    res = minimize_gradient_descent(
        quadratic, quadratic_jac, np.array([5.0, -3, 1]), max_iterations=2
    )
    assert not res.converged
    assert res.n_iterations == 2


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        minimize_gradient_descent(quadratic, quadratic_jac, np.zeros(3), 2)  # ty: ignore[too-many-positional-arguments]


def test_result_is_immutable():
    res = minimize_gradient_descent(quadratic, quadratic_jac, np.array([5.0, -3, 1]))
    with pytest.raises(AttributeError):
        res.fun = 1.0  # ty: ignore[invalid-assignment]


def test_repr():
    res = minimize_gradient_descent(quadratic, quadratic_jac, np.array([5.0, -3, 1]))
    assert repr(res).startswith("GradientDescentResult(fun=")


class MyError(Exception):
    pass


def _raise_my_error(x):
    raise MyError("from user function")


def test_exception_in_fun_is_reraised_unchanged():
    with pytest.raises(MyError, match="from user function"):
        minimize_gradient_descent(_raise_my_error, quadratic_jac, np.zeros(3))


def test_exception_in_jac_is_reraised_unchanged():
    with pytest.raises(MyError, match="from user function"):
        minimize_gradient_descent(quadratic, _raise_my_error, np.zeros(3))


def test_fun_returning_non_number_raises_type_error():
    with pytest.raises(TypeError):
        minimize_gradient_descent(lambda x: "a", quadratic_jac, np.zeros(3))  # ty: ignore[invalid-argument-type]


def test_jac_with_wrong_length_raises_value_error():
    with pytest.raises(ValueError, match="jac must return an array of length 3"):
        minimize_gradient_descent(quadratic, lambda x: np.ones(2), np.zeros(3))


def test_jac_with_nan_raises_value_error():
    with pytest.raises(ValueError, match="non-finite"):
        minimize_gradient_descent(quadratic, lambda x: np.full(3, np.nan), np.zeros(3))


@pytest.mark.parametrize(
    "options",
    [
        {"gradient_tolerance": -1.0},
        {"initial_step_size": 0.0},
        {"shrink_factor": 1.5},
        {"sufficient_decrease": 0.0},
        {"max_line_search_iterations": 0},
    ],
)
def test_invalid_options_raise_value_error(options):
    with pytest.raises(ValueError):
        minimize_gradient_descent(quadratic, quadratic_jac, np.zeros(3), **options)


def test_negative_max_iterations_raises_overflow_error():
    with pytest.raises(OverflowError):
        minimize_gradient_descent(
            quadratic, quadratic_jac, np.zeros(3), max_iterations=-1
        )
