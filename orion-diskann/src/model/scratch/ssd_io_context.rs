/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![allow(dead_code)]

#[cfg(target_family = "unix")]
use memmap2::Mmap;
#[cfg(target_os = "windows")]
use orion_ann_platform::{FileHandle, IOCompletionPort};
#[cfg(target_family = "unix")]
use std::os::fd::RawFd;
use std::sync::Arc;
// Todo: Remove this when the disk index query code is complete.
use crate::common::ANNError;

/// Legacy read lifecycle state, including a retained I/O failure.
pub enum Status {
    ReadWait,
    ReadSuccess,
    ReadFailed(ANNError),
    ProcessComplete,
}

// The IOContext struct for disk I/O. One for each thread.
/// Per-worker Windows file and completion-port handles with read lifecycle state.
#[cfg(target_os = "windows")]
pub struct IOContext {
    /// Current read/completion state of this I/O context.
    pub status: Status,

    /// Windows file handle used by aligned reads.
    pub file_handle: FileHandle,

    /// Windows completion queue receiving asynchronous I/O outcomes.
    pub io_completion_port: IOCompletionPort,
}

#[cfg(target_os = "windows")]
impl Default for IOContext {
    /// Creates an unregistered Windows I/O context in its waiting state.
    fn default() -> Self {
        IOContext {
            status: Status::ReadWait,
            file_handle: FileHandle::default(),
            io_completion_port: IOCompletionPort::default(),
        }
    }
}

#[cfg(target_os = "windows")]
impl IOContext {
    /// Creates the default unregistered Windows I/O context.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Unix I/O context replacing Windows IOContext (FileHandle + IOCompletionPort).
/// Uses RawFd + optional Mmap for memory-mapped file access.
#[cfg(target_family = "unix")]
pub struct UnixIOContext {
    /// File descriptor supplied to the read adapter.
    pub fd: RawFd,

    /// Optional mapped view of the opened file.
    pub mmap: Option<Arc<Mmap>>,

    /// Current read/completion state of this I/O context.
    pub status: orion_ann_platform::Status,
}

#[cfg(target_family = "unix")]
impl UnixIOContext {
    /// Creates a Unix context for an existing descriptor without a mapped view.
    pub fn new(fd: RawFd) -> Self {
        UnixIOContext {
            fd,
            mmap: None,
            status: orion_ann_platform::Status::ReadWait,
        }
    }

    /// Creates a ready Unix context that shares the supplied read-only mapping.
    pub fn with_mmap(fd: RawFd, mmap: Arc<Mmap>) -> Self {
        UnixIOContext {
            fd,
            mmap: Some(mmap),
            status: orion_ann_platform::Status::ReadSuccess,
        }
    }
}

#[cfg(target_family = "unix")]
impl Default for UnixIOContext {
    /// Creates an unregistered Unix I/O context with no file or mapping.
    fn default() -> Self {
        UnixIOContext {
            fd: -1,
            mmap: None,
            status: orion_ann_platform::Status::ReadWait,
        }
    }
}
