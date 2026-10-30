use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArrayLike1};
use optimagic_rust::gradient_descent as rs;
use optimagic_rust::line_search::BacktrackingOptions;
use pyo3::prelude::*;

use crate::callbacks::{wrap_fun, wrap_jac};
use crate::errors::to_py_err;

/// Result of gradient descent.
#[pyclass(frozen, get_all, module = "optimagic_rust")]
pub struct GradientDescentResult {
    /// Best parameter vector found, shape (n,).
    x: Py<PyArray1<f64>>,
    /// Objective value at ``x``.
    fun: f64,
    /// Number of iterations, i.e. accepted steps.
    n_iterations: usize,
    /// Number of objective evaluations.
    n_fun_evals: usize,
    /// Number of gradient evaluations.
    n_jac_evals: usize,
    /// Whether the gradient norm reached ``gradient_tolerance``.
    converged: bool,
    /// Why the algorithm stopped.
    message: String,
}

#[pymethods]
impl GradientDescentResult {
    fn __repr__(&self) -> String {
        format!(
            "GradientDescentResult(fun={}, n_iterations={}, converged={}, message={:?})",
            self.fun,
            self.n_iterations,
            if self.converged { "True" } else { "False" },
            self.message,
        )
    }
}

impl GradientDescentResult {
    fn new(py: Python<'_>, res: rs::GradientDescentResult) -> Self {
        Self {
            x: res.x.into_pyarray(py).unbind(),
            fun: res.fun,
            n_iterations: res.n_iterations,
            n_fun_evals: res.n_fun_evals,
            n_jac_evals: res.n_jac_evals,
            converged: res.converged,
            message: res.message,
        }
    }
}

/// Minimize a function with gradient descent and a backtracking line search.
///
/// In each iteration, the algorithm moves along the negative gradient. The step size is
/// determined by :func:`backtracking_line_search`. The algorithm stops when the
/// Euclidean norm of the gradient is at most ``gradient_tolerance`` (converged), when
/// the line search fails or after ``max_iterations`` steps.
///
/// This is a naive algorithm that converges slowly on badly scaled problems.
///
/// Args:
///     fun: Objective function. Takes a 1d numpy array and returns a float.
///     jac: Gradient of ``fun``. Takes a 1d numpy array and returns an array of the
///         same shape.
///     x0: Start parameters, shape (n,).
///     max_iterations: Maximum number of iterations, i.e. accepted steps.
///     gradient_tolerance: Stop when the Euclidean norm of the gradient is at most this
///         value. Must be non-negative.
///     initial_step_size: First step size the line search tries in each iteration.
///     shrink_factor: Factor by which the line search shrinks the step size after each
///         rejected trial.
///     sufficient_decrease: Constant of the Armijo condition in the line search.
///     max_line_search_iterations: Maximum number of objective evaluations per line
///         search.
///
/// Returns:
///     The optimization result.
///
/// Raises:
///     ValueError: If an option is invalid, or ``jac`` returns an array of the wrong
///         length or with non-finite entries.
#[pyfunction]
#[pyo3(signature = (
    fun,
    jac,
    x0,
    *,
    max_iterations = 1000,
    gradient_tolerance = 1e-6,
    initial_step_size = 1.0,
    shrink_factor = 0.5,
    sufficient_decrease = 1e-4,
    max_line_search_iterations = 50,
))]
#[allow(clippy::too_many_arguments)]
pub fn minimize_gradient_descent(
    py: Python<'_>,
    fun: &Bound<'_, PyAny>,
    jac: &Bound<'_, PyAny>,
    x0: PyArrayLike1<f64, AllowTypeChange>,
    max_iterations: usize,
    gradient_tolerance: f64,
    initial_step_size: f64,
    shrink_factor: f64,
    sufficient_decrease: f64,
    max_line_search_iterations: usize,
) -> PyResult<GradientDescentResult> {
    let options = rs::GradientDescentOptions {
        max_iterations,
        gradient_tolerance,
        line_search: BacktrackingOptions {
            initial_step_size,
            shrink_factor,
            sufficient_decrease,
            max_iterations: max_line_search_iterations,
        },
    };
    let res = rs::minimize_gradient_descent(wrap_fun(fun), wrap_jac(jac), x0.as_array(), &options)
        .map_err(to_py_err)?;
    Ok(GradientDescentResult::new(py, res))
}
