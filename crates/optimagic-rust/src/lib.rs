//! Low-level building blocks for numerical optimizers.
//!
//! This crate is pure Rust and knows nothing about Python. The Python bindings live in
//! the `optimagic-rust-py` crate.
//!
//! All algorithms take user-provided functions (objective, gradient) as closures that
//! return a `Result`. If a closure fails, the algorithm stops and returns the closure's
//! error unchanged inside [`OptimizationError::Callback`].
//!
//! # Example
//!
//! ```
//! use ndarray::{Array1, ArrayView1, array};
//! use optimagic_rust::gradient_descent::{GradientDescentOptions, minimize_gradient_descent};
//! use std::convert::Infallible;
//!
//! // f(x) = x'x has its minimum at the origin.
//! let fun = |x: ArrayView1<f64>| Ok::<f64, Infallible>(x.dot(&x));
//! let jac = |x: ArrayView1<f64>| Ok::<Array1<f64>, Infallible>(2.0 * &x);
//!
//! let options = GradientDescentOptions::default();
//! let res = minimize_gradient_descent(fun, jac, array![1.0, -2.0].view(), &options).unwrap();
//!
//! assert!(res.converged);
//! assert!(res.x.iter().all(|xi| xi.abs() < 1e-6));
//! ```

mod error;
pub mod gradient_descent;
pub mod line_search;

pub use error::OptimizationError;
