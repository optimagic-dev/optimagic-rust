use optimagic_rust::OptimizationError;
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Convert an error of the Rust core into a Python exception.
///
/// Exceptions raised by user-provided Python functions are re-raised unchanged, so
/// users see their original exception and traceback.
pub(crate) fn to_py_err(err: OptimizationError<PyErr>) -> PyErr {
    match err {
        OptimizationError::InvalidInput(msg) => PyValueError::new_err(msg),
        OptimizationError::Callback(err) => err,
    }
}
