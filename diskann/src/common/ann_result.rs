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

#[derive(thiserror::Error, Debug)]
pub enum ANNError {
    #[error("IndexError: {err}")]
    IndexError { err: String },

    #[error("IndexConfigError: parameter={parameter}, err={err}")]
    IndexConfigError { parameter: String, err: String },

    #[error("TryFromIntError: {err}")]
    TryFromIntError {
        #[from]
        err: TryFromIntError,
    },

    #[error("IOError: {err}")]
    IOError {
        #[from]
        err: io::Error,
    },

    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayoutError {
        #[from]
        err: LayoutError,
    },

    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    #[error("DiskIOAlignmentError: {err}")]
    DiskIOAlignmentError { err: String },

    #[error("LogError: {err}")]
    LogError {
        #[from]
        err: LogError,
    },

    #[error("PQError: {err}")]
    PQError { err: String },

    #[error("Error try creating array from slice: {err}")]
    TryFromSliceError {
        #[from]
        err: TryFromSliceError,
    },

    #[error("CandidateSetsError: {err}")]
    CandidateSetsError { err: String },

    #[error("SerializeError: {err}")]
    SerializeError {
        #[from]
        err: bincode::error::EncodeError,
    },

    #[error("DeserializeError: {err}")]
    DeserializeError {
        #[from]
        err: bincode::error::DecodeError,
    },

    #[error("ClusterError: {err}")]
    ClusterError { err: String },

    #[cfg(all(feature = "staged_diskann", feature = "visualization"))]
    #[error("VisualizationError: {err}")]
    VisualizationError { err: String },
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
    pub fn log_candidate_sets_error(err: String) -> Self {
        let ann_err = ANNError::CandidateSetsError { err };
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

    #[inline]
    pub fn log_cluster_error(err: String) -> Self {
        let ann_err = ANNError::ClusterError { err };
        match log_error(ann_err.to_string()) {
            Ok(()) => ann_err,
            Err(log_err) => ANNError::LogError { err: log_err },
        }
    }

    #[cfg(all(feature = "staged_diskann", feature = "visualization"))]
    #[inline]
    pub fn log_visualization_error(err: String) -> Self {
        let ann_err = ANNError::VisualizationError { err };
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
