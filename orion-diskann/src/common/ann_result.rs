/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_logger::LogError;
use std::alloc::LayoutError;
use std::array::TryFromSliceError;
use std::io;
use std::num::TryFromIntError;

pub type ANNResult<T> = Result<T, ANNError>;

/// DiskANN's internal error type. Construction and conversion are side-effect
/// free; applications report errors at their outer boundary. Legacy `log_*`
/// names remain for source compatibility, but no longer invoke the logger.
#[derive(thiserror::Error, Debug)]
pub enum ANNError {
    #[error("IndexError: {err}")]
    IndexError { err: String },

    #[error("IndexConfigError: parameter={parameter}, err={err}")]
    IndexConfigError { parameter: String, err: String },

    #[error("TryFromIntError: {err}")]
    TryFromIntError {
        #[source]
        err: TryFromIntError,
    },

    #[error("IOError: {err}")]
    IOError {
        #[source]
        err: io::Error,
    },

    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayoutError {
        #[source]
        err: LayoutError,
    },

    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    #[error("DiskIOAlignmentError: {err}")]
    DiskIOAlignmentError { err: String },

    #[error("LogError: {err}")]
    LogError {
        #[source]
        err: LogError,
    },

    #[error("PQError: {err}")]
    PQError { err: String },

    #[error("Error try creating array from slice: {err}")]
    TryFromSliceError {
        #[source]
        err: TryFromSliceError,
    },

    #[error("SerializeError: {err}")]
    SerializeError {
        #[source]
        err: bincode::error::EncodeError,
    },

    #[error("DeserializeError: {err}")]
    DeserializeError {
        #[source]
        err: bincode::error::DecodeError,
    },
}

impl From<io::Error> for ANNError {
    #[inline]
    fn from(err: io::Error) -> Self {
        Self::log_io_error(err)
    }
}

impl From<TryFromIntError> for ANNError {
    #[inline]
    fn from(err: TryFromIntError) -> Self {
        Self::log_try_from_int_error(err)
    }
}

impl From<LayoutError> for ANNError {
    #[inline]
    fn from(err: LayoutError) -> Self {
        Self::log_mem_alloc_layout_error(err)
    }
}

impl From<TryFromSliceError> for ANNError {
    #[inline]
    fn from(err: TryFromSliceError) -> Self {
        Self::log_try_from_slice_error(err)
    }
}

impl From<bincode::error::EncodeError> for ANNError {
    #[inline]
    fn from(err: bincode::error::EncodeError) -> Self {
        Self::log_serialize_error(err)
    }
}

impl From<bincode::error::DecodeError> for ANNError {
    #[inline]
    fn from(err: bincode::error::DecodeError) -> Self {
        Self::log_deserialize_error(err)
    }
}

// Retain explicit logger failures without replacing unrelated errors.
impl From<LogError> for ANNError {
    #[inline]
    fn from(err: LogError) -> Self {
        Self::LogError { err }
    }
}

impl ANNError {
    #[inline]
    pub fn log_index_error(err: String) -> Self {
        ANNError::IndexError { err }
    }

    #[inline]
    pub fn log_index_config_error(parameter: String, err: String) -> Self {
        ANNError::IndexConfigError { parameter, err }
    }

    #[inline]
    pub fn log_try_from_int_error(err: TryFromIntError) -> Self {
        ANNError::TryFromIntError { err }
    }

    #[inline]
    pub fn log_io_error(err: io::Error) -> Self {
        ANNError::IOError { err }
    }

    #[inline]
    pub fn log_disk_io_request_alignment_error(err: String) -> Self {
        ANNError::DiskIOAlignmentError { err }
    }

    #[inline]
    pub fn log_mem_alloc_layout_error(err: LayoutError) -> Self {
        ANNError::MemoryAllocLayoutError { err }
    }

    #[inline]
    pub fn log_lock_poison_error(err: String) -> Self {
        ANNError::LockPoisonError { err }
    }

    #[inline]
    pub fn log_pq_error(err: String) -> Self {
        ANNError::PQError { err }
    }

    #[inline]
    pub fn log_try_from_slice_error(err: TryFromSliceError) -> Self {
        ANNError::TryFromSliceError { err }
    }

    #[inline]
    pub fn log_serialize_error(err: bincode::error::EncodeError) -> Self {
        ANNError::SerializeError { err }
    }

    #[inline]
    pub fn log_deserialize_error(err: bincode::error::DecodeError) -> Self {
        ANNError::DeserializeError { err }
    }
}

#[cfg(test)]
mod ann_result_test {
    use super::*;

    #[test]
    fn ann_err_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ANNError>();
    }
}

// The source backend owns this optional conversion; the public error crate
// remains independent of DiskANN and its platform dependencies.
#[cfg(feature = "orion")]
mod public_error {
    use orion_ann_error::{Error, ErrorCode};
    use serde_json::json;
    impl From<super::ANNError> for Error {
        fn from(source: super::ANNError) -> Self {
            use super::ANNError as A;
            let code = match &source {
                A::IndexConfigError { .. } => ErrorCode::InvalidArgument,
                A::IOError { err }
                    if matches!(
                        err.kind(),
                        std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof
                    ) =>
                {
                    ErrorCode::InvalidData
                }
                A::IOError { .. } => ErrorCode::Io,
                A::LockPoisonError { .. } => ErrorCode::InvalidState,
                A::MemoryAllocLayoutError { .. } => ErrorCode::ResourceExhausted,
                A::DeserializeError { .. } | A::TryFromSliceError { .. } => ErrorCode::InvalidData,
                _ => ErrorCode::Backend,
            };
            let message = match &source {
                A::IndexError { err }
                | A::IndexConfigError { err, .. }
                | A::LockPoisonError { err }
                | A::DiskIOAlignmentError { err }
                | A::PQError { err } => err.clone(),
                _ => std::error::Error::source(&source)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| source.to_string()),
            };
            let mut error = Self::new(code, message);
            match &source {
                A::IndexConfigError { parameter, .. } => {
                    error = error.with_detail("field", parameter.clone());
                }
                A::IOError { err } => {
                    error = error
                        .with_detail("kind", format!("{:?}", err.kind()))
                        .with_detail("os_error", json!(err.raw_os_error()));
                }
                _ => {}
            }
            error.with_source(source)
        }
    }
}
