//! Wrap Python callables as Rust closures that the core algorithms can call.

use ndarray::{Array1, ArrayView1};
use numpy::{AllowTypeChange, PyArrayLike1, ToPyArray};
use pyo3::prelude::*;

/// Wrap a Python objective function `fun(x: ndarray) -> float`.
///
/// Each call passes a fresh copy of `x` to Python, so the callable may keep or modify
/// its argument without affecting the algorithm.
pub(crate) fn wrap_fun<'py>(
    fun: &Bound<'py, PyAny>,
) -> impl FnMut(ArrayView1<f64>) -> PyResult<f64> {
    move |x| fun.call1((x.to_pyarray(fun.py()),))?.extract()
}

/// Wrap a Python gradient function `jac(x: ndarray) -> array_like`.
///
/// The return value may be anything numpy can convert to a 1d float array.
pub(crate) fn wrap_jac<'py>(
    jac: &Bound<'py, PyAny>,
) -> impl FnMut(ArrayView1<f64>) -> PyResult<Array1<f64>> {
    move |x| {
        let grad = jac.call1((x.to_pyarray(jac.py()),))?;
        let grad: PyArrayLike1<f64, AllowTypeChange> = grad.extract()?;
        Ok(grad.as_array().to_owned())
    }
}
