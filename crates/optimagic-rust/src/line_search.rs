//! Backtracking line search with the Armijo sufficient decrease condition.

use ndarray::ArrayView1;

use crate::OptimizationError;

/// Options for [`backtracking_line_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct BacktrackingOptions {
    /// First step size that is tried. Must be positive.
    pub initial_step_size: f64,
    /// Factor by which the step size shrinks after each rejected trial. Must be in
    /// (0, 1).
    pub shrink_factor: f64,
    /// Constant `c` of the Armijo condition. Must be in (0, 1).
    pub sufficient_decrease: f64,
    /// Maximum number of objective evaluations. Must be at least 1.
    pub max_iterations: usize,
}

impl Default for BacktrackingOptions {
    fn default() -> Self {
        Self {
            initial_step_size: 1.0,
            shrink_factor: 0.5,
            sufficient_decrease: 1e-4,
            max_iterations: 50,
        }
    }
}

impl BacktrackingOptions {
    pub(crate) fn validate<E>(&self) -> Result<(), OptimizationError<E>> {
        let invalid = |msg: String| Err(OptimizationError::InvalidInput(msg));
        // The negated comparisons also reject NaN.
        if !(self.initial_step_size > 0.0 && self.initial_step_size.is_finite()) {
            return invalid(format!(
                "initial_step_size must be positive and finite, but is {}.",
                self.initial_step_size
            ));
        }
        if !(self.shrink_factor > 0.0 && self.shrink_factor < 1.0) {
            return invalid(format!(
                "shrink_factor must be in (0, 1), but is {}.",
                self.shrink_factor
            ));
        }
        if !(self.sufficient_decrease > 0.0 && self.sufficient_decrease < 1.0) {
            return invalid(format!(
                "sufficient_decrease must be in (0, 1), but is {}.",
                self.sufficient_decrease
            ));
        }
        if self.max_iterations == 0 {
            return invalid("max_iterations must be at least 1, but is 0.".to_string());
        }
        Ok(())
    }
}

/// Result of [`backtracking_line_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct LineSearchResult {
    /// Accepted step size, or the last tried step size if `converged` is false.
    pub step_size: f64,
    /// Objective value at `x + step_size * direction`.
    pub fun: f64,
    /// Number of objective evaluations.
    pub n_fun_evals: usize,
    /// Whether the Armijo condition is satisfied at `step_size`.
    pub converged: bool,
}

/// Find a step size along `direction` that sufficiently decreases `fun`.
///
/// Starting from `options.initial_step_size`, the step size `t` is multiplied by
/// `options.shrink_factor` until the Armijo condition
///
/// ```text
/// fun(x + t * direction) <= fun_x + c * t * grad_x' direction
/// ```
///
/// holds, where `c` is `options.sufficient_decrease`, or until
/// `options.max_iterations` step sizes were tried.
///
/// # Arguments
///
/// * `fun` - Objective function.
/// * `x` - Current point, shape (n,).
/// * `fun_x` - Objective value at `x`.
/// * `grad_x` - Gradient of the objective at `x`, shape (n,).
/// * `direction` - Search direction, shape (n,). Must be a descent direction, i.e.
///   `grad_x' direction < 0`.
/// * `options` - Tuning parameters.
///
/// # Errors
///
/// Returns [`OptimizationError::InvalidInput`] if the options are invalid, the arrays
/// have different lengths or `direction` is not a descent direction, and
/// [`OptimizationError::Callback`] if `fun` fails.
pub fn backtracking_line_search<F, E>(
    mut fun: F,
    x: ArrayView1<f64>,
    fun_x: f64,
    grad_x: ArrayView1<f64>,
    direction: ArrayView1<f64>,
    options: &BacktrackingOptions,
) -> Result<LineSearchResult, OptimizationError<E>>
where
    F: FnMut(ArrayView1<f64>) -> Result<f64, E>,
{
    options.validate()?;
    if grad_x.len() != x.len() || direction.len() != x.len() {
        return Err(OptimizationError::InvalidInput(format!(
            "x, grad_x and direction must have the same length, but have lengths {}, {} \
             and {}.",
            x.len(),
            grad_x.len(),
            direction.len()
        )));
    }
    let slope = grad_x.dot(&direction);
    if slope.is_nan() || slope >= 0.0 {
        return Err(OptimizationError::InvalidInput(format!(
            "direction must be a descent direction, i.e. grad_x' direction must be \
             negative, but it is {slope}."
        )));
    }

    let mut step_size = options.initial_step_size;
    let mut n_fun_evals = 0;
    loop {
        let mut trial = x.to_owned();
        trial.scaled_add(step_size, &direction);
        let fun_trial = fun(trial.view()).map_err(OptimizationError::Callback)?;
        n_fun_evals += 1;

        // NaN never satisfies the condition, so the step size keeps shrinking.
        let converged = fun_trial <= fun_x + options.sufficient_decrease * step_size * slope;
        if converged || n_fun_evals >= options.max_iterations {
            return Ok(LineSearchResult {
                step_size,
                fun: fun_trial,
                n_fun_evals,
                converged,
            });
        }
        step_size *= options.shrink_factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;
    use std::convert::Infallible;

    fn sphere(x: ArrayView1<f64>) -> Result<f64, Infallible> {
        Ok(x.dot(&x))
    }

    #[test]
    fn test_full_step_is_accepted_if_it_decreases_enough() {
        // A step of 0.5 along -1 moves from x = 1 to x = 0.5, where x^2 = 0.25.
        let options = BacktrackingOptions {
            initial_step_size: 0.5,
            ..Default::default()
        };
        let x = array![1.0];
        let direction = array![-1.0];
        let res = backtracking_line_search(
            sphere,
            x.view(),
            1.0,
            array![2.0].view(),
            direction.view(),
            &options,
        )
        .unwrap();
        assert!(res.converged);
        assert_eq!(res.n_fun_evals, 1);
        assert_relative_eq!(res.step_size, 0.5);
        assert_relative_eq!(res.fun, 0.25);
    }

    #[test]
    fn test_step_shrinks_until_armijo_condition_holds() {
        // A step of 1 along -2 jumps from x = 1 to x = -1, which does not decrease x^2.
        // The second trial (step 0.5) reaches the minimum.
        let res = backtracking_line_search(
            sphere,
            array![1.0].view(),
            1.0,
            array![2.0].view(),
            array![-2.0].view(),
            &BacktrackingOptions::default(),
        )
        .unwrap();
        assert!(res.converged);
        assert_eq!(res.n_fun_evals, 2);
        assert_relative_eq!(res.step_size, 0.5);
        assert_relative_eq!(res.fun, 0.0);
    }

    #[test]
    fn test_not_converged_after_max_iterations() {
        let options = BacktrackingOptions {
            max_iterations: 3,
            ..Default::default()
        };
        let res = backtracking_line_search(
            |_x: ArrayView1<f64>| Ok::<f64, Infallible>(f64::INFINITY),
            array![1.0].view(),
            1.0,
            array![2.0].view(),
            array![-2.0].view(),
            &options,
        )
        .unwrap();
        assert!(!res.converged);
        assert_eq!(res.n_fun_evals, 3);
        assert_relative_eq!(res.step_size, 0.25);
    }

    #[test]
    fn test_callback_error_is_returned_unchanged() {
        let res = backtracking_line_search(
            |_x: ArrayView1<f64>| Err::<f64, &str>("boom"),
            array![1.0].view(),
            1.0,
            array![2.0].view(),
            array![-2.0].view(),
            &BacktrackingOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::Callback("boom"))));
    }

    #[test]
    fn test_ascent_direction_is_rejected() {
        let res = backtracking_line_search(
            sphere,
            array![1.0].view(),
            1.0,
            array![2.0].view(),
            array![2.0].view(),
            &BacktrackingOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::InvalidInput(_))));
    }

    #[test]
    fn test_length_mismatch_is_rejected() {
        let res = backtracking_line_search(
            sphere,
            array![1.0, 2.0].view(),
            5.0,
            array![2.0].view(),
            array![-2.0, -4.0].view(),
            &BacktrackingOptions::default(),
        );
        assert!(matches!(res, Err(OptimizationError::InvalidInput(_))));
    }

    #[test]
    fn test_invalid_options_are_rejected() {
        let invalid = [
            BacktrackingOptions {
                initial_step_size: 0.0,
                ..Default::default()
            },
            BacktrackingOptions {
                initial_step_size: f64::NAN,
                ..Default::default()
            },
            BacktrackingOptions {
                shrink_factor: 1.0,
                ..Default::default()
            },
            BacktrackingOptions {
                sufficient_decrease: 0.0,
                ..Default::default()
            },
            BacktrackingOptions {
                max_iterations: 0,
                ..Default::default()
            },
        ];
        for options in invalid {
            assert!(matches!(
                options.validate::<Infallible>(),
                Err(OptimizationError::InvalidInput(_))
            ));
        }
    }
}
