use numpy::{AllowTypeChange, PyArrayLike1};
use optimagic_rust::line_search as rs;
use pyo3::prelude::*;

use crate::callbacks::wrap_fun;
use crate::errors::to_py_err;

/// Result of a backtracking line search.
#[pyclass(frozen, get_all, module = "optimagic_rust")]
pub struct LineSearchResult {
    /// Accepted step size, or the last tried step size if not converged.
    step_size: f64,
    /// Objective value at ``x + step_size * direction``.
    fun: f64,
    /// Number of objective evaluations.
    n_fun_evals: usize,
    /// Whether the Armijo condition is satisfied at ``step_size``.
    converged: bool,
}

#[pymethods]
impl LineSearchResult {
    fn __repr__(&self) -> String {
        format!(
            "LineSearchResult(step_size={}, fun={}, n_fun_evals={}, converged={})",
            self.step_size,
            self.fun,
            self.n_fun_evals,
            if self.converged { "True" } else { "False" },
        )
    }
}

impl From<rs::LineSearchResult> for LineSearchResult {
    fn from(res: rs::LineSearchResult) -> Self {
        Self {
            step_size: res.step_size,
            fun: res.fun,
            n_fun_evals: res.n_fun_evals,
            converged: res.converged,
        }
    }
}

/// Find a step size along ``direction`` that sufficiently decreases ``fun``.
///
/// Starting from ``initial_step_size``, the step size ``t`` is multiplied by
/// ``shrink_factor`` until the Armijo condition
/// ``fun(x + t * direction) <= fun_x + sufficient_decrease * t * grad_x @ direction``
/// holds, or until ``max_iterations`` step sizes were tried.
///
/// Args:
///     fun: Objective function. Takes a 1d numpy array and returns a float.
///     x: Current point, shape (n,).
///     fun_x: Objective value at ``x``.
///     grad_x: Gradient of the objective at ``x``, shape (n,).
///     direction: Search direction, shape (n,). Must be a descent direction, i.e.
///         ``grad_x @ direction < 0``.
///     initial_step_size: First step size that is tried. Must be positive.
///     shrink_factor: Factor by which the step size shrinks after each rejected trial.
///         Must be in (0, 1).
///     sufficient_decrease: Constant of the Armijo condition. Must be in (0, 1).
///     max_iterations: Maximum number of objective evaluations. Must be at least 1.
///
/// Returns:
///     The result of the line search.
///
/// Raises:
///     ValueError: If an option is invalid, the arrays have different lengths or
///         ``direction`` is not a descent direction.
#[pyfunction]
#[pyo3(signature = (
    fun,
    x,
    fun_x,
    grad_x,
    direction,
    *,
    initial_step_size = 1.0,
    shrink_factor = 0.5,
    sufficient_decrease = 1e-4,
    max_iterations = 50,
))]
#[allow(clippy::too_many_arguments)]
pub fn backtracking_line_search(
    fun: &Bound<'_, PyAny>,
    x: PyArrayLike1<f64, AllowTypeChange>,
    fun_x: f64,
    grad_x: PyArrayLike1<f64, AllowTypeChange>,
    direction: PyArrayLike1<f64, AllowTypeChange>,
    initial_step_size: f64,
    shrink_factor: f64,
    sufficient_decrease: f64,
    max_iterations: usize,
) -> PyResult<LineSearchResult> {
    let options = rs::BacktrackingOptions {
        initial_step_size,
        shrink_factor,
        sufficient_decrease,
        max_iterations,
    };
    let res = rs::backtracking_line_search(
        wrap_fun(fun),
        x.as_array(),
        fun_x,
        grad_x.as_array(),
        direction.as_array(),
        &options,
    )
    .map_err(to_py_err)?;
    Ok(res.into())
}
