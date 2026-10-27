//! Best-effort operating-system hints for large resident allocations.

/// Align large allocations to the usual Linux PMD huge-page boundary.
/// This is a layout hint, not a request for reserved hugetlb pages.
#[cfg(target_os = "linux")]
pub const HUGE_PAGE_ALIGNMENT: usize = 2 * 1024 * 1024;

/// Ask Linux to consider complete pages inside an allocation for THP.
///
/// Call before initializing the allocation so first-touch faults can use large
/// pages. Success means the advice was accepted, not that THP was allocated.
/// The caller may continue using ordinary pages if this returns an error.
///
/// # Safety
/// `ptr..ptr+bytes` must be a live, writable allocation exclusively owned by
/// the caller. Only pages wholly contained in that allocation are advised.
#[cfg(target_os = "linux")]
pub unsafe fn advise_huge_pages(ptr: *mut u8, bytes: usize) -> std::io::Result<()> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(std::io::Error::other("could not determine OS page size"));
    }
    let page_size = page_size as usize;
    let leading = (page_size - ptr.addr() % page_size) % page_size;
    if leading >= bytes {
        return Ok(());
    }
    let length = (bytes - leading) / page_size * page_size;
    if length == 0 {
        return Ok(());
    }
    // Exclude partial end pages: madvise must not change allocator metadata
    // or another allocation sharing one of those pages.
    let result = unsafe { libc::madvise(ptr.add(leading).cast(), length, libc::MADV_HUGEPAGE) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
