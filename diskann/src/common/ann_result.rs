/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use logger::LogError;
use logger::error_logger::log_error;
use std::alloc::LayoutError;
use std::array::TryFromSliceError;
use std::io;
use std::num::TryFromIntError;

pub type ANNResult<T> = Result<T, ANNError>;

/// Unified error type for the workspace.
///
/// Every `From<T>` impl below routes through the matching
/// `log_*_error` helper, so a plain `?` propagation **automatically
/// logs at the point of conversion** — no `.map_err(...)` boilerplate
/// at call sites. New error sources should:
///   1. Add a variant here (no `#[from]` — we write the From manually).
///   2. Add a `log_*_error` helper on the `impl ANNError` block.
///   3. Add a `From<T> for ANNError` impl that calls the helper.
///
/// See `From<io::Error>` below for the canonical pattern.
#[derive(thiserror::Error, Debug)]
pub enum ANNError {
    #[error("IndexError: {err}")]
    IndexError { err: String },

    #[error("IndexConfigError: parameter={parameter}, err={err}")]
    IndexConfigError { parameter: String, err: String },

    #[error("TryFromIntError: {err}")]
    TryFromIntError { err: TryFromIntError },

    #[error("IOError: {err}")]
    IOError { err: io::Error },

    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayoutError { err: LayoutError },

    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    #[error("DiskIOAlignmentError: {err}")]
    DiskIOAlignmentError { err: String },

    #[error("LogError: {err}")]
    LogError { err: LogError },

    #[error("PQError: {err}")]
    PQError { err: String },

    #[error("Error try creating array from slice: {err}")]
    TryFromSliceError { err: TryFromSliceError },

    #[error("SerializeError: {err}")]
    SerializeError { err: bincode::error::EncodeError },

    #[error("DeserializeError: {err}")]
    DeserializeError { err: bincode::error::DecodeError },
}

// ── Auto-logging `From` impls ───────────────────────────────────────────
//
// `?` propagation calls `From::from(err)`. By routing each From through
// the corresponding `log_*_error` helper, every conversion point emits
// a log line — no need to sprinkle `.map_err(ANNError::log_*_error)` at
// every call site. `thiserror`'s `#[from]` derive would skip the log
// step, so we hand-roll these.

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

// `LogError`'s `From` is deliberately NOT auto-logging — recursing into
// the logger on a logger failure would loop. Wrap manually if needed.
impl From<LogError> for ANNError {
    #[inline]
    fn from(err: LogError) -> Self {
        Self::LogError { err }
    }
}

impl ANNError {
    #[inline]
    pub fn log_index_error(err: String) -> Self {
        let ann_err = ANNError::IndexError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_index_config_error(parameter: String, err: String) -> Self {
        let ann_err = ANNError::IndexConfigError { parameter, err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_try_from_int_error(err: TryFromIntError) -> Self {
        let ann_err = ANNError::TryFromIntError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_io_error(err: io::Error) -> Self {
        let ann_err = ANNError::IOError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_disk_io_request_alignment_error(err: String) -> Self {
        let ann_err = ANNError::DiskIOAlignmentError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_mem_alloc_layout_error(err: LayoutError) -> Self {
        let ann_err = ANNError::MemoryAllocLayoutError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_lock_poison_error(err: String) -> Self {
        let ann_err = ANNError::LockPoisonError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_pq_error(err: String) -> Self {
        let ann_err = ANNError::PQError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_try_from_slice_error(err: TryFromSliceError) -> Self {
        let ann_err = ANNError::TryFromSliceError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_serialize_error(err: bincode::error::EncodeError) -> Self {
        let ann_err = ANNError::SerializeError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[inline]
    pub fn log_deserialize_error(err: bincode::error::DecodeError) -> Self {
        let ann_err = ANNError::DeserializeError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
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
