# Type stub for the compiled extension module optimagic_rust._core.
#
# Type checkers and IDEs cannot look inside compiled code, so this file declares its
# API. Keep it in sync with crates/optimagic-rust-py/src:
# - `pixi run -e type-checking stubtest` checks names and signatures.
# - tests/test_stubs.py checks that docstrings here equal the Rust doc comments.

from collections.abc import Callable
from typing import final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GradientDescentResult",
    "LineSearchResult",
    "__version__",
    "backtracking_line_search",
    "minimize_gradient_descent",
]

__version__: str

@final
class LineSearchResult:
    """Result of a backtracking line search."""

    @property
    def step_size(self) -> float:
        """Accepted step size, or the last tried step size if not converged."""

    @property
    def fun(self) -> float:
        """Objective value at ``x + step_size * direction``."""

    @property
    def n_fun_evals(self) -> int:
        """Number of objective evaluations."""

    @property
    def converged(self) -> bool:
        """Whether the Armijo condition is satisfied at ``step_size``."""

@final
class GradientDescentResult:
    """Result of gradient descent."""

    @property
    def x(self) -> NDArray[np.float64]:
        """Best parameter vector found, shape (n,)."""

    @property
    def fun(self) -> float:
        """Objective value at ``x``."""

    @property
    def n_iterations(self) -> int:
        """Number of iterations, i.e. accepted steps."""

    @property
    def n_fun_evals(self) -> int:
        """Number of objective evaluations."""

    @property
    def n_jac_evals(self) -> int:
        """Number of gradient evaluations."""

    @property
    def converged(self) -> bool:
        """Whether the gradient norm reached ``gradient_tolerance``."""

    @property
    def message(self) -> str:
        """Why the algorithm stopped."""

def backtracking_line_search(
    fun: Callable[[NDArray[np.float64]], float],
    x: ArrayLike,
    fun_x: float,
    grad_x: ArrayLike,
    direction: ArrayLike,
    *,
    initial_step_size: float = 1.0,
    shrink_factor: float = 0.5,
    sufficient_decrease: float = 1e-4,
    max_iterations: int = 50,
) -> LineSearchResult:
    """Find a step size along ``direction`` that sufficiently decreases ``fun``.

    Starting from ``initial_step_size``, the step size ``t`` is multiplied by
    ``shrink_factor`` until the Armijo condition
    ``fun(x + t * direction) <= fun_x + sufficient_decrease * t * grad_x @ direction``
    holds, or until ``max_iterations`` step sizes were tried.

    Args:
        fun: Objective function. Takes a 1d numpy array and returns a float.
        x: Current point, shape (n,).
        fun_x: Objective value at ``x``.
        grad_x: Gradient of the objective at ``x``, shape (n,).
        direction: Search direction, shape (n,). Must be a descent direction, i.e.
            ``grad_x @ direction < 0``.
        initial_step_size: First step size that is tried. Must be positive.
        shrink_factor: Factor by which the step size shrinks after each rejected trial.
            Must be in (0, 1).
        sufficient_decrease: Constant of the Armijo condition. Must be in (0, 1).
        max_iterations: Maximum number of objective evaluations. Must be at least 1.

    Returns:
        The result of the line search.

    Raises:
        ValueError: If an option is invalid, the arrays have different lengths or
            ``direction`` is not a descent direction.

    """

def minimize_gradient_descent(
    fun: Callable[[NDArray[np.float64]], float],
    jac: Callable[[NDArray[np.float64]], ArrayLike],
    x0: ArrayLike,
    *,
    max_iterations: int = 1000,
    gradient_tolerance: float = 1e-6,
    initial_step_size: float = 1.0,
    shrink_factor: float = 0.5,
    sufficient_decrease: float = 1e-4,
    max_line_search_iterations: int = 50,
) -> GradientDescentResult:
    """Minimize a function with gradient descent and a backtracking line search.

    In each iteration, the algorithm moves along the negative gradient. The step size is
    determined by :func:`backtracking_line_search`. The algorithm stops when the
    Euclidean norm of the gradient is at most ``gradient_tolerance`` (converged), when
    the line search fails or after ``max_iterations`` steps.

    This is a naive algorithm that converges slowly on badly scaled problems.

    Args:
        fun: Objective function. Takes a 1d numpy array and returns a float.
        jac: Gradient of ``fun``. Takes a 1d numpy array and returns an array of the
            same shape.
        x0: Start parameters, shape (n,).
        max_iterations: Maximum number of iterations, i.e. accepted steps.
        gradient_tolerance: Stop when the Euclidean norm of the gradient is at most this
            value. Must be non-negative.
        initial_step_size: First step size the line search tries in each iteration.
        shrink_factor: Factor by which the line search shrinks the step size after each
            rejected trial.
        sufficient_decrease: Constant of the Armijo condition in the line search.
        max_line_search_iterations: Maximum number of objective evaluations per line
            search.

    Returns:
        The optimization result.

    Raises:
        ValueError: If an option is invalid, or ``jac`` returns an array of the wrong
            length or with non-finite entries.

    """
