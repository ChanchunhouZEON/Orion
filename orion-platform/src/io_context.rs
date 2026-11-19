/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use memmap2::Mmap;
use std::os::unix::io::RawFd;

/// Status of an I/O operation.
pub enum Status {
    ReadWait,
    ReadSuccess,
    ReadFailed(std::io::Error),
    ProcessComplete,
}

/// Unix I/O context replacing Windows IOContext (FileHandle + IOCompletionPort).
/// Uses RawFd + optional Mmap for memory-mapped file access.
pub struct UnixIOContext {
    /// File descriptor supplied to the read adapter.
    pub fd: RawFd,

    /// Optional mapped view of the opened file.
    pub mmap: Option<Mmap>,

    /// Current read/completion state of this I/O context.
    pub status: Status,
}

impl UnixIOContext {
    /// Creates an unread file-descriptor context without mapping the file.
    pub fn new(fd: RawFd) -> Self {
        UnixIOContext {
            fd,
            mmap: None,
            status: Status::ReadWait,
        }
    }

    /// Creates a successful read context that retains the supplied mapping.
    pub fn with_mmap(fd: RawFd, mmap: Mmap) -> Self {
        UnixIOContext {
            fd,
            mmap: Some(mmap),
            status: Status::ReadSuccess,
        }
    }
}

impl Default for UnixIOContext {
    /// Creates an unregistered context with no usable descriptor or mapping.
    fn default() -> Self {
        UnixIOContext {
            fd: -1,
            mmap: None,
            status: Status::ReadWait,
        }
    }
}
