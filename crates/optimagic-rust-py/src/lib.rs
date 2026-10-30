//! Python bindings for the `optimagic-rust` crate.
//!
//! The code here only converts between Python and Rust types and calls the pure-Rust
//! implementations. It contains no algorithmic logic. maturin compiles this crate into
//! the extension module `optimagic_rust._core`.
//!
//! Every function or class exported here also needs an entry in the type stub
//! `python/optimagic_rust/_core.pyi` and a re-export in
//! `python/optimagic_rust/__init__.py`.

use pyo3::prelude::*;

mod callbacks;
mod errors;
mod gradient_descent;
mod line_search;

/// Rust implementations of low-level components for numerical optimizers.
#[pymodule(name = "_core")]
mod optimagic_rust_core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::gradient_descent::{GradientDescentResult, minimize_gradient_descent};
    #[pymodule_export]
    use crate::line_search::{LineSearchResult, backtracking_line_search};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
