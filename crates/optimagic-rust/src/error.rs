/// Error returned by the algorithms in this crate.
///
/// `E` is the error type of the user-provided functions (objective, gradient). The
/// algorithms never inspect it; they hand it back unchanged, so that callers (e.g. the
/// Python bindings) can re-raise the original error.
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError<E> {
    /// An argument, or the output of a user-provided function, has an invalid value or
    /// shape.
    #[error("{0}")]
    InvalidInput(String),
    /// A user-provided function returned an error.
    #[error("user-provided function failed: {0}")]
    Callback(E),
}
