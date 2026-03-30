/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

/// Access mode for file operations.
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// Unix file handle wrapping std::fs::File.
/// Replaces Windows FileHandle (CreateFileA/CloseHandle).
pub struct UnixFileHandle {
    file: File,
}

impl UnixFileHandle {
    /// Open a file with the specified access mode.
    pub fn open<P: AsRef<Path>>(path: P, mode: AccessMode) -> io::Result<Self> {
        let file = match mode {
            AccessMode::Read => File::open(path)?,
            AccessMode::Write => OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?,
            AccessMode::ReadWrite => OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)?,
        };
        Ok(UnixFileHandle { file })
    }

    /// Get the underlying File reference.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Get the raw file descriptor.
    pub fn raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Get the file size in bytes.
    pub fn file_size(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

impl Default for UnixFileHandle {
    fn default() -> Self {
        // Create a handle to /dev/null as a default placeholder
        UnixFileHandle {
            file: File::open("/dev/null").unwrap_or_else(|_| {
                // Fallback: create a temp file
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open("/tmp/.diskann_null")
                    .expect("Cannot open fallback file")
            }),
        }
    }
}
