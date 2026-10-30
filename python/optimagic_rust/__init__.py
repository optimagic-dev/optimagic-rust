"""Rust implementations of low-level components for numerical optimizers.

Everything is implemented in the compiled extension module ``optimagic_rust._core``
and re-exported here. The type stub ``_core.pyi`` describes its API.

"""

from optimagic_rust._core import (
    GradientDescentResult,
    LineSearchResult,
    __version__,
    backtracking_line_search,
    minimize_gradient_descent,
)

__all__ = [
    "GradientDescentResult",
    "LineSearchResult",
    "__version__",
    "backtracking_line_search",
    "minimize_gradient_descent",
]
