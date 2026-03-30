/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::io;
use std::os::unix::io::RawFd;

use crate::aligned_reader::DISK_IO_ALIGNMENT;

/// Aligned read request, analogous to the Windows OVERLAPPED structure.
pub struct AlignedRead<T: Default + Clone> {
    pub offset: u64,
    pub len: usize,
    pub buf: Vec<T>,
}

impl<T: Default + Clone> AlignedRead<T> {
    /// Create a new aligned read request.
    pub fn new(offset: u64, len: usize) -> Self {
        AlignedRead {
            offset,
            len,
            buf: vec![T::default(); len],
        }
    }

    /// Create a new aligned read with a pre-allocated buffer.
    pub fn with_buf(offset: u64, buf: Vec<T>) -> Self {
        let len = buf.len();
        AlignedRead { offset, len, buf }
    }
}

/// Perform a pread-based fallback read (when mmap is not suitable).
/// Uses libc::pread for positional I/O without seeking.
pub fn pread_aligned(fd: RawFd, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let ret = unsafe {
        libc::pread(
            fd,
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
            offset as libc::off_t,
        )
    };

    if ret < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(ret as usize)
    }
}

/// Check if a value is aligned to the disk I/O alignment boundary.
#[inline]
pub fn is_sector_aligned(val: u64) -> bool {
    val % (DISK_IO_ALIGNMENT as u64) == 0
}

/// Round up a value to the next sector-aligned boundary.
#[inline]
pub fn sector_align_up(val: u64) -> u64 {
    let align = DISK_IO_ALIGNMENT as u64;
    (val + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sector_alignment() {
        assert!(is_sector_aligned(0));
        assert!(is_sector_aligned(512));
        assert!(is_sector_aligned(1024));
        assert!(!is_sector_aligned(1));
        assert!(!is_sector_aligned(513));
    }

    #[test]
    fn test_sector_align_up() {
        assert_eq!(sector_align_up(0), 0);
        assert_eq!(sector_align_up(1), 512);
        assert_eq!(sector_align_up(512), 512);
        assert_eq!(sector_align_up(513), 1024);
        assert_eq!(sector_align_up(1023), 1024);
        assert_eq!(sector_align_up(1024), 1024);
    }
}
