/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! The only conversion boundary between Orion errors and Python exceptions.
use orion_ann::{Error, Result};
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

pyo3::create_exception!(_native, OrionError, pyo3::exceptions::PyRuntimeError);

pub fn to_python(py: Python<'_>, error: Error) -> PyErr {
    fn build(py: Python<'_>, error: Error) -> PyResult<PyErr> {
        let exception = OrionError::new_err(error.to_string());
        let report = serde_json::to_string(&error.report())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let fields = py.import_bound("json")?.call_method1("loads", (report,))?;
        let value = exception.value_bound(py);
        for field in ["code", "operation", "message", "details", "causes"] {
            value.setattr(field, fields.get_item(field)?)?;
        }
        value.setattr("report", fields)?;
        Ok(exception)
    }
    build(py, error).unwrap_or_else(|error| error)
}

pub fn result<T>(py: Python<'_>, value: Result<T>) -> PyResult<T> {
    value.map_err(|error| to_python(py, error))
}

/// Catch only unforeseen unwinding failures. Normal failures use `Result`.
/// The closure never publishes a partially constructed object. Existing Index
/// mutations hold their exclusive write lock inside the closure, so an unwind poisons the
/// handle and future calls reject it rather than using partially mutated state.
/// Python's process-wide panic hook is deliberately left untouched.
pub fn boundary<T>(
    py: Python<'_>,
    operation: &str,
    f: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic payload");
            Err(to_python(
                py,
                Error::internal("unexpected Rust panic; reopen the index before retrying")
                    .at(operation)
                    .with_detail("reason", reason)
                    .with_detail("requires_reopen", true),
            ))
        }
    }
}

pub fn poisoned() -> Error {
    Error::invalid_state("index operation panicked; reopen the index")
        .with_detail("requires_reopen", true)
}
