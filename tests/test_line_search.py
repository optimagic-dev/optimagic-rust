"""Tests for the Python bindings of the backtracking line search.

The algorithm itself is tested in Rust (crates/optimagic-rust/src/line_search.rs).

"""

import numpy as np
import pytest

from optimagic_rust import LineSearchResult, backtracking_line_search


def sphere(x):
    return float(x @ x)


def test_step_shrinks_until_armijo_condition_holds():
    # A step of 1 along -2 jumps from x = 1 to x = -1, which does not decrease x^2.
    # The second trial (step 0.5) reaches the minimum.
    res = backtracking_line_search(
        sphere, x=[1.0], fun_x=1.0, grad_x=[2.0], direction=[-2.0]
    )
    assert isinstance(res, LineSearchResult)
    assert res.converged
    assert res.step_size == 0.5
    assert res.fun == 0.0
    assert res.n_fun_evals == 2


def test_not_converged_after_max_iterations():
    res = backtracking_line_search(
        lambda x: np.inf,
        x=np.array([1.0]),
        fun_x=1.0,
        grad_x=np.array([2.0]),
        direction=np.array([-2.0]),
        max_iterations=3,
    )
    assert not res.converged
    assert res.n_fun_evals == 3
    assert res.step_size == 0.25


def test_ascent_direction_raises_value_error():
    with pytest.raises(ValueError, match="descent direction"):
        backtracking_line_search(
            sphere, x=[1.0], fun_x=1.0, grad_x=[2.0], direction=[2.0]
        )


def test_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="same length"):
        backtracking_line_search(
            sphere, x=[1.0, 2.0], fun_x=5.0, grad_x=[2.0], direction=[-2.0, -4.0]
        )


def test_exception_in_fun_is_reraised_unchanged():
    class MyError(Exception):
        pass

    def fun(x):
        raise MyError("from user function")

    with pytest.raises(MyError, match="from user function"):
        backtracking_line_search(fun, x=[1.0], fun_x=1.0, grad_x=[2.0], direction=[-2])


def test_repr():
    res = backtracking_line_search(
        sphere, x=[1.0], fun_x=1.0, grad_x=[2.0], direction=[-2.0]
    )
    assert repr(res) == (
        "LineSearchResult(step_size=0.5, fun=0, n_fun_evals=2, converged=True)"
    )
