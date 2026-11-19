/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Platform-independent error wire types shared by native and WASM frontends.
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// Stable machine-readable categories. Match codes, not human-readable text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// Returns the stable `ORION_*` identifier shared by Rust, JSON and FFI reports.
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

/// Display uses the same stable code as JSON/FFI reports.
impl std::fmt::Display for ErrorCode {
    /// Writes the stable wire identifier rather than the Rust variant name.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for ErrorCode {
    /// Serializes the stable wire identifier so adapters share one error vocabulary.
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
    /// Stable machine-readable category; applications should branch on this field.
    pub code: &'static str,

    /// Closest failing operation, when an adapter or subsystem supplied one.
    pub operation: Option<String>,

    /// Human-facing explanation; its wording is not a stable matching contract.
    pub message: String,

    /// Structured failure context, with keys specific to the failing operation.
    pub details: BTreeMap<String, Value>,

    /// Underlying error messages ordered from the immediate cause outward.
    pub causes: Vec<String>,
}
