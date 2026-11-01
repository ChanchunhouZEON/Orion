//! Stable errors shared by the Rust API, command line and language bindings.
//!
//! Constructing or converting an error never logs. Applications decide where
//! to report failures; the original source remains available for diagnostics.
use serde_json::{Value, json};
use std::{collections::BTreeMap, fmt, path::Path};

mod report;
pub use report::{ErrorCode, ErrorReport};

/// Public error with structured context and a preserved source chain.
///
/// Successful calls allocate no diagnostic data. Details are constructed only
/// on failure, outside the distance and prefetch loops.
#[derive(Debug)]
pub struct Error {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    code: ErrorCode,
    operation: Option<String>,
    message: String,
    details: BTreeMap<String, Value>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

/// The result returned by Orion's public operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// Classify a worker-pool construction failure without coupling this crate to Rayon.
    pub fn thread_pool(source: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::new(ErrorCode::ResourceExhausted, source.to_string())
            .at("create_thread_pool")
            .with_source(source)
    }

    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                code,
                operation: None,
                message: message.into(),
                details: BTreeMap::new(),
                source: None,
            }),
        }
    }
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.inner.message = message.into();
        self
    }
    pub fn code(&self) -> ErrorCode {
        self.inner.code
    }
    pub fn operation(&self) -> Option<&str> {
        self.inner.operation.as_deref()
    }
    pub fn message(&self) -> &str {
        &self.inner.message
    }
    pub fn details(&self) -> &BTreeMap<String, Value> {
        &self.inner.details
    }
    /// Attach the closest operation; outer adapters do not overwrite it.
    pub fn at(mut self, operation: impl Into<String>) -> Self {
        if self.inner.operation.is_none() {
            self.inner.operation = Some(operation.into());
        }
        self
    }
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.inner.details.insert(key.into(), value.into());
        self
    }
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.inner.source = Some(Box::new(source));
        self
    }
    pub fn configuration(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::invalid_argument(message).with_detail("field", field.into())
    }
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidData, message)
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UnsupportedConfiguration, message)
    }
    pub fn cache(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::CacheIncompatible, message)
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidState, message)
    }
    pub fn dimension(field: &str, expected: usize, actual: usize) -> Self {
        Self::new(
            ErrorCode::DimensionMismatch,
            format!("{field} must be {expected}, got {actual}"),
        )
        .with_detail("field", field)
        .with_detail("expected", expected)
        .with_detail("actual", actual)
    }
    /// Validate a packed row-major matrix without guessing a row dimension.
    pub fn packed_shape(field: &str, dimension: usize, elements: usize) -> Self {
        Self::new(
            ErrorCode::DimensionMismatch,
            format!("{field} element count {elements} must be divisible by dimension {dimension}"),
        )
        .with_detail("field", field)
        .with_detail("dimension", dimension)
        .with_detail("elements", elements)
    }
    pub fn io(operation: &str, path: &Path, source: std::io::Error) -> Self {
        Self::from(source)
            .at(operation)
            .with_detail("path", path.to_string_lossy().into_owned())
    }
    pub fn report(&self) -> ErrorReport {
        let mut causes = Vec::new();
        let mut source = std::error::Error::source(self);
        while let Some(error) = source {
            causes.push(error.to_string());
            source = error.source();
        }
        ErrorReport {
            code: self.inner.code.as_str(),
            operation: self.inner.operation.clone(),
            message: self.inner.message.clone(),
            details: self.inner.details.clone(),
            causes,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.inner.code.as_str())?;
        if let Some(operation) = &self.inner.operation {
            write!(f, "{operation}: ")?;
        }
        write!(f, "{}", self.inner.message)?;
        if let Some(path) = self.inner.details.get("path").and_then(Value::as_str) {
            write!(f, " (path: {path})")?;
        }
        Ok(())
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.inner
            .source
            .as_ref()
            .map(|s| s.as_ref() as &(dyn std::error::Error + 'static))
    }
}
// Legacy validation helpers in vector/DiskANN still return descriptive strings.
// These adapters are for validation only; I/O and backend errors use typed From.
impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::invalid_argument(message)
    }
}
impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::invalid_argument(message)
    }
}
impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        let kind = format!("{:?}", source.kind());
        let errno = source.raw_os_error();
        Self::new(
            if matches!(
                source.kind(),
                std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
            ) {
                ErrorCode::InvalidData
            } else {
                ErrorCode::Io
            },
            source.to_string(),
        )
        .with_detail("kind", kind)
        .with_detail("os_error", json!(errno))
        .with_source(source)
    }
}
impl From<serde_json::Error> for Error {
    fn from(source: serde_json::Error) -> Self {
        let code = if source.is_io() {
            ErrorCode::Io
        } else {
            ErrorCode::InvalidData
        };
        Self::new(code, source.to_string())
            .with_detail("line", source.line())
            .with_detail("column", source.column())
            .with_source(source)
    }
}

impl From<std::time::SystemTimeError> for Error {
    fn from(source: std::time::SystemTimeError) -> Self {
        Self::invalid_data(source.to_string()).with_source(source)
    }
}

impl From<bincode::error::EncodeError> for Error {
    fn from(source: bincode::error::EncodeError) -> Self {
        let code = if matches!(source, bincode::error::EncodeError::Io { .. }) {
            ErrorCode::Io
        } else {
            ErrorCode::Internal
        };
        Self::new(code, source.to_string()).with_source(source)
    }
}
impl From<bincode::error::DecodeError> for Error {
    fn from(source: bincode::error::DecodeError) -> Self {
        let code = if matches!(source, bincode::error::DecodeError::Io { .. }) {
            ErrorCode::Io
        } else {
            ErrorCode::InvalidData
        };
        Self::new(code, source.to_string()).with_source(source)
    }
}

impl From<std::num::TryFromIntError> for Error {
    fn from(source: std::num::TryFromIntError) -> Self {
        Self::invalid_data(source.to_string()).with_source(source)
    }
}
