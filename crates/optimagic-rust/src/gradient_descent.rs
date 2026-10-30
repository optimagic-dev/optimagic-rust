//! Gradient descent with a backtracking line search.
//!
//! This is a deliberately naive algorithm. It exists to exercise the toolchain (Rust
//! crate, Python bindings, packaging) and converges slowly on badly scaled problems.

use ndarray::{Array1, ArrayView1};

use crate::OptimizationError;
use crate::line_search::{BacktrackingOptions, backtracking_line_search};

/// Options for [`minimize_gradient_descent`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradientDescentOptions {
    /// Maximum number of iterations, i.e. accepted steps.
    pub max_iterations: usize,
    /// Stop when the Euclidean norm of the gradient is at most this value. Must be
    /// non-negative.
    pub gradient_tolerance: f64,
    /// Options of the line search that determines the step size in each iteration.
    pub line_search: BacktrackingOptions,
}

impl Default for GradientDescentOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gradient_tolerance: 1e-6,
            line_search: BacktrackingOptions::default(),
        }
    }
}

impl GradientDescentOptions {
    fn validate<E>(&self) -> Result<(), OptimizationError<E>> {
        if self.gradient_tolerance.is_nan() || self.gradient_tolerance < 0.0 {
            return Err(OptimizationError::InvalidInput(format!(
                "gradient_tolerance must be non-negative, but is {}.",
                self.gradient_tolerance
            )));
        }
        self.line_search.validate()
    }
}

/// Result of [`minimize_gradient_descent`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradientDescentResult {
    /// Best parameter vector found, shape (n,).
    pub x: Array1<f64>,
    /// Objective value at `x`.
    pub fun: f64,
    /// Number of iterations, i.e. accepted steps.
    pub n_iterations: usize,
    /// Number of objective evaluations.
    pub n_fun_evals: usize,
    /// Number of gradient evaluations.
    pub n_jac_evals: usize,
    /// Whether the gradient norm reached `gradient_tolerance`.
    pub converged: bool,
    /// Why the algorithm stopped.
    pub message: String,
}

/// Minimize `fun` with gradient descent.
///
/// In each iteration, the algorithm moves along the negative gradient. The step size is
/// determined by [`backtracking_line_search`]. The algorithm stops when
///
/// - the Euclidean norm of the gradient is at most `options.gradient_tolerance`
///   (converged),
/// - the line search does not find a step size that decreases the objective
///   sufficiently, or
/// - `options.max_iterations` steps were taken.
///
/// # Arguments
///
/// * `fun` - Objective function.
/// * `jac` - Gradient of the objective function. Must return an array of shape (n,).
/// * `x0` - Start parameters, shape (n,).
/// * `options` - Tuning parameters.
///
/// # Errors
///
/// Returns [`OptimizationError::InvalidInput`] if the options are invalid or `jac`
/// returns an array of the wrong length or with non-finite entries, and
/// [`OptimizationError::Callback`] if `fun` or `jac` fails.
pub fn minimize_gradient_descent<F, G, E>(
    mut fun: F,
    mut jac: G,
    x0: ArrayView1<f64>,
    options: &GradientDescentOptions,
) -> Result<GradientDescentResult, OptimizationError<E>>
where
    F: FnMut(ArrayView1<f64>) -> Result<f64, E>,
    G: FnMut(ArrayView1<f64>) -> Result<Array1<f64>, E>,
{
    options.validate()?;

    let mut x = x0.to_owned();
    let mut fun_x = fun(x.view()).map_err(OptimizationError::Callback)?;
    let mut n_fun_evals = 1;
    let mut n_jac_evals = 0;
    let mut n_iterations = 0;

    let (converged, message) = loop {
        let grad = jac(x.view()).map_err(OptimizationError::Callback)?;
        n_jac_evals += 1;
        check_gradient(&grad, x.len())?;

        if grad.dot(&grad).sqrt() <= options.gradient_tolerance {
            break (true, "The gradient norm is below gradient_tolerance.");
        }
        if n_iterations >= options.max_iterations {
            break (false, "Reached max_iterations.");
        }

        let direction = -&grad;
        let ls = backtracking_line_search(
            &mut fun,
            x.view(),
            fun_x,
            grad.view(),
            direction.view(),
            &options.line_search,
        )?;
        n_fun_evals += ls.n_fun_evals;
        if !ls.converged {
            break (
                false,
                "The line search did not find a step size that decreases the objective \
                 sufficiently.",
            );
        }
        x.scaled_add(ls.step_size, &direction);
        fun_x = ls.fun;
        n_iterations += 1;
    };

    Ok(GradientDescentResult {
        x,
        fun: fun_x,
        n_iterations,
        n_fun_evals,
        n_jac_evals,
        converged,
        message: message.to_string(),
    })
}

fn check_gradient<E>(grad: &Array1<f64>, n_params: usize) -> Result<(), OptimizationError<E>> {
    if grad.len() != n_params {
        return Err(OptimizationError::InvalidInput(format!(
            "jac must return an array of length {n_params}, but returned one of length {}.",
            grad.len()
        )));
    }
    if !grad.iter().all(|g| g.is_finite()) {
        return Err(OptimizationError::InvalidInput(format!(
            "jac returned non-finite values: {grad}."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::convert::Infallible;

    /// f(x) = sum_i w_i (x_i - i)^2 with weights w = (1, 2, 3).
    fn quadratic(x: ArrayView1<f64>) -> Result<f64, Infallible> {
        Ok((0..3)
            .map(|i| (i + 1) as f64 * (x[i] - i as f64).powi(2))
            .sum())
    }

    fn quadratic_jac(x: ArrayView1<f64>) -> Result<Array1<f64>, Infallible> {
        Ok(Array1::from_shape_fn(3, |i| {
            2.0 * (i + 1) as f64 * (x[i] - i as f64)
        }))
    }

    #[test]
    fn test_converges_on_quadratic() {
        let x0 = array![5.0, -3.0, 1.0];
        let res = minimize_gradient_descent(
            quadratic,
            quadratic_jac,
            x0.view(),
            &GradientDescentOptions::default(),
        )
        .unwrap();
        assert!(res.converged, "{}", res.message);
        assert_abs_diff_eq!(res.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(res.x[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(res.x[2], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(res.fun, 0.0, epsilon = 1e-10);
        assert!(res.n_fun_evals > res.n_iterations);
        assert_eq!(res.n_jac_evals, res.n_iterations + 1);
    }

    #[test]
    fn test_start_at_optimum_takes_no_step() {
        let x0 = array![0.0, 1.0, 2.0];
        let res = minimize_gradient_descent(
            quadratic,
            quadratic_jac,
            x0.view(),
            &GradientDescentOptions::default(),
        )
        .unwrap();
        assert!(res.converged);
        assert_eq!(res.n_iterations, 0);
        assert_eq!(res.n_fun_evals, 1);
        assert_eq!(res.x, x0);
    }

    #[test]
    fn test_stops_at_max_iterations() {
        let options = GradientDescentOptions {
            max_iterations: 2,
            ..Default::default()
        };
        let res = minimize_gradient_descent(
            quadratic,
            quadratic_jac,
            array![5.0, -3.0, 1.0].view(),
            &options,
        )
        .unwrap();
        assert!(!res.converged);
        assert_eq!(res.n_iterations, 2);
        assert_eq!(res.message, "Reached max_iterations.");
    }

    #[test]
    fn test_wrong_gradient_length_is_rejected() {
        let res = minimize_gradient_descent(
            quadratic,
            |_x: ArrayView1<f64>| Ok::<Array1<f64>, Infallible>(array![1.0]),
            array![5.0, -3.0, 1.0].view(),
            &GradientDescentOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::InvalidInput(_))));
    }

    #[test]
    fn test_non_finite_gradient_is_rejected() {
        let res = minimize_gradient_descent(
            quadratic,
            |_x: ArrayView1<f64>| Ok::<Array1<f64>, Infallible>(array![1.0, f64::NAN, 1.0]),
            array![5.0, -3.0, 1.0].view(),
            &GradientDescentOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::InvalidInput(_))));
    }

    #[test]
    fn test_callback_error_is_returned_unchanged() {
        let res = minimize_gradient_descent(
            |_x: ArrayView1<f64>| Err::<f64, &str>("boom"),
            |x: ArrayView1<f64>| Ok(x.to_owned()),
            array![1.0].view(),
            &GradientDescentOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::Callback("boom"))));
    }

    #[test]
    fn test_negative_gradient_tolerance_is_rejected() {
        let options = GradientDescentOptions {
            gradient_tolerance: -1.0,
            ..Default::default()
        };
        let res = minimize_gradient_descent(
            quadratic,
            quadratic_jac,
            array![5.0, -3.0, 1.0].view(),
            &options,
        );
        assert!(matches!(res, Err(OptimizationError::InvalidInput(_))));
    }
}
