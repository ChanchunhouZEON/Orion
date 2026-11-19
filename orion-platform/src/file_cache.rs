/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Writeback and cache invalidation for completed private file ranges.
//! These operations never change file contents or allocation accounting. Success
//! means the OS accepted the request, not that a measured number of pages was freed.

use std::{fs::File, io, ops::Range};

/// Maximum completed construction window before requesting writeback/reclamation.
/// This bounds dirty output between checkpoints, not RSS or total page-cache use.
pub use crate::buffered_io::RECLAIM_WINDOW_BYTES;

/// Persist completed file bytes and request eviction of their cached pages.
///
/// Use on private staging output, after flushing any userspace writer. The range
/// is in file bytes. Boundary pages may also be affected; concurrent users can
/// immediately fault them back in. This is not a global cache flush or an RSS cap.
///
/// # Errors
/// Rejects reversed/out-of-file ranges and propagates writeback or advice errors.
/// Unsupported operating systems return `Unsupported`; empty ranges are no-ops.
pub fn sync_and_reclaim_file(file: &File, range: Range<u64>) -> io::Result<()> {
    if range.start > range.end || range.end > file.metadata()?.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "file cache range out of bounds",
        ));
    }
    if range.is_empty() {
        return Ok(());
    }
    file.sync_data()?;
    #[cfg(target_os = "linux")]
    {
        advise_file(file, range)
    }
    #[cfg(target_os = "macos")]
    {
        // The temporary view is never dereferenced. The kernel validates msync;
        // no borrowed Rust slice is exposed even if an external file changes.
        let page = page_size()? as u64;
        let start = range.start / page * page;
        let len = usize::try_from(range.end - start).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "file cache range too large")
        })?;
        let map = unsafe {
            memmap2::MmapOptions::new()
                .offset(start)
                .len(len)
                .map(file)?
        };
        // SAFETY: live file-backed view; the address is page-aligned and bounded.
        unsafe { sync_mapping(map.as_ptr(), map.len()) }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "file cache reclamation requires Linux or macOS",
        ))
    }
}

/// Flush and invalidate an owned shared mapping, then discard its Linux file cache.
///
/// # Safety
/// `ptr..ptr+len` must be a live, page-aligned MAP_SHARED mapping of `file` at
/// offset zero, with no concurrent mutation. Never pass heap or COW memory.
pub(crate) unsafe fn reclaim_mapping(
    ptr: *const u8,
    len: usize,
    file: &File,
    range: Range<usize>,
) -> io::Result<()> {
    if range.start > range.end || range.end > len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "mapping cache range out of bounds",
        ));
    }
    if range.is_empty() {
        return Ok(());
    }
    let page = page_size()?;
    let start = range.start / page * page;
    // SAFETY: caller retains the backing, and the checked range stays in it.
    unsafe {
        sync_mapping(ptr.add(start), range.end - start)?;
    }
    #[cfg(target_os = "linux")]
    {
        // Drop this mapping's PTE references before asking the page cache to
        // release clean pages. Neither operation changes the shared file bytes.
        let result = unsafe {
            libc::madvise(
                ptr.add(start).cast_mut().cast(),
                range.end - start,
                libc::MADV_DONTNEED,
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        advise_file(file, start as u64..range.end as u64)?;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
    Ok(())
}

/// Returns the actual VM page size; cache lines and huge-page hints are unrelated.
fn page_size() -> io::Result<usize> {
    #[cfg(unix)]
    {
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size > 0 {
            return Ok(size as usize);
        }
        Err(io::Error::last_os_error())
    }
    #[cfg(not(unix))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "page size unavailable",
        ))
    }
}

/// Persist shared file pages; Darwin also invalidates their cached contents.
///
/// # Safety
/// The address must be page-aligned within a live shared file mapping, and len
/// must be nonzero and bounded by that mapping. No concurrent mutation is allowed.
unsafe fn sync_mapping(ptr: *const u8, len: usize) -> io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        #[cfg(target_os = "linux")]
        let flags = libc::MS_SYNC;
        #[cfg(target_os = "macos")]
        let flags = libc::MS_SYNC | libc::MS_INVALIDATE;
        if unsafe { libc::msync(ptr.cast_mut().cast(), len, flags) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (ptr, len);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "mapping cache reclamation requires Linux or macOS",
        ))
    }
}

/// Submit whole-page file advice, including the page containing a final short tail.
#[cfg(target_os = "linux")]
fn advise_file(file: &File, range: Range<u64>) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let page = page_size()? as u64;
    let start = range.start / page * page;
    let end = range
        .end
        .checked_add(page - 1)
        .map(|n| n / page * page)
        .filter(|n| *n <= i64::MAX as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "file advice range overflow"))?;
    let code = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            start as libc::off_t,
            (end - start) as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    // posix_fadvise returns errno directly, unlike madvise/msync.
    if code != 0 {
        return Err(io::Error::from_raw_os_error(code));
    }
    Ok(())
}
