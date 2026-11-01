//! Platform-independent error wire types shared by native and WASM frontends.
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Stable machine-readable categories. Match codes, not human-readable text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    InvalidArgument,
    DimensionMismatch,
    UnsupportedConfiguration,
    InvalidData,
    CacheIncompatible,
    ResourceExhausted,
    Io,
    InvalidState,
    Backend,
    Internal,
}
impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgument => "ORION_INVALID_ARGUMENT",
            Self::DimensionMismatch => "ORION_DIMENSION_MISMATCH",
            Self::UnsupportedConfiguration => "ORION_UNSUPPORTED_CONFIGURATION",
            Self::InvalidData => "ORION_INVALID_DATA",
            Self::CacheIncompatible => "ORION_CACHE_INCOMPATIBLE",
            Self::ResourceExhausted => "ORION_RESOURCE_EXHAUSTED",
            Self::Io => "ORION_IO",
            Self::InvalidState => "ORION_INVALID_STATE",
            Self::Backend => "ORION_BACKEND",
            Self::Internal => "ORION_INTERNAL",
        }
    }
}

impl Serialize for ErrorCode {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// A language-neutral report. FFI adapters transport this instead of inventing
/// their own messages. OS-specific causes are retained separately in `causes`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorReport {
    pub code: &'static str,
    pub operation: Option<String>,
    pub message: String,
    pub details: BTreeMap<String, Value>,
    pub causes: Vec<String>,
}
