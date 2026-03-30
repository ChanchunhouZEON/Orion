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
use platform::{FileHandle, IOCompletionPort};
#[cfg(target_family = "unix")]
use std::os::fd::RawFd;
use std::sync::Arc;
// Todo: Remove this when the disk index query code is complete.
use crate::common::ANNError;

pub enum Status {
    ReadWait,
    ReadSuccess,
    ReadFailed(ANNError),
    ProcessComplete,
}

// The IOContext struct for disk I/O. One for each thread.
#[cfg(target_os = "windows")]
pub struct IOContext {
    pub status: Status,
    pub file_handle: FileHandle,
    pub io_completion_port: IOCompletionPort,
}

#[cfg(target_os = "windows")]
impl Default for IOContext {
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
    pub fn new() -> Self {
        Self::default()
    }
}

/// Unix I/O context replacing Windows IOContext (FileHandle + IOCompletionPort).
/// Uses RawFd + optional Mmap for memory-mapped file access.
#[cfg(target_family = "unix")]
pub struct UnixIOContext {
    pub fd: RawFd,
    pub mmap: Option<Arc<Mmap>>,
    pub status: platform::Status,
}

#[cfg(target_family = "unix")]
impl UnixIOContext {
    pub fn new(fd: RawFd) -> Self {
        UnixIOContext {
            fd,
            mmap: None,
            status: platform::Status::ReadWait,
        }
    }

    pub fn with_mmap(fd: RawFd, mmap: Arc<Mmap>) -> Self {
        UnixIOContext {
            fd,
            mmap: Some(mmap),
            status: platform::Status::ReadSuccess,
        }
    }
}

#[cfg(target_family = "unix")]
impl Default for UnixIOContext {
    fn default() -> Self {
        UnixIOContext {
            fd: -1,
            mmap: None,
            status: platform::Status::ReadWait,
        }
    }
}
