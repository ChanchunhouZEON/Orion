/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Scoped RAM residency, independent of search stages and CPU cache hints.
//!
//! Guards borrow their allocations. Overlapping guards share OS page locks:
//! `mlock` itself is not reference counted. Only locks acquired through this
//! module participate in that accounting; do not mix external `munlock` calls
//! with these guards. This does not pin data in L1/L2 or request huge pages.
use std::{
    io,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    users: usize,
}

#[derive(Debug, Default)]
struct Registry {
    spans: Vec<Span>,
}

impl Registry {
    // Split only at allocation boundaries, not into one entry per OS page.
    // Metadata therefore stays proportional to the number of pinned regions.
    fn split(&mut self, at: usize) {
        if let Some(i) = self.spans.iter().position(|s| s.start < at && at < s.end) {
            let s = self.spans[i];
            self.spans[i].end = at;
            self.spans.insert(i + 1, Span { start: at, ..s });
        }
    }
    // Releasing a short-lived overlapping guard must not retain its boundaries
    // forever in a long-lived guard's metadata.
    fn coalesce(&mut self) {
        let mut write = 0;
        for read in 0..self.spans.len() {
            let span = self.spans[read];
            if write > 0
                && self.spans[write - 1].end == span.start
                && self.spans[write - 1].users == span.users
            {
                self.spans[write - 1].end = span.end;
            } else {
                self.spans[write] = span;
                write += 1;
            }
        }
        self.spans.truncate(write);
    }
    fn acquire(
        &mut self,
        start: usize,
        end: usize,
        mut lock: impl FnMut(usize, usize) -> io::Result<()>,
        mut unlock: impl FnMut(usize, usize),
    ) -> io::Result<()> {
        self.split(start);
        self.split(end);
        let mut cursor = start;
        let mut gaps = Vec::new();
        for s in self.spans.iter().filter(|s| s.end > start && s.start < end) {
            if cursor < s.start {
                gaps.push((cursor, s.start));
            }
            cursor = s.end;
        }
        if cursor < end {
            gaps.push((cursor, end));
        }
        for (i, &(a, b)) in gaps.iter().enumerate() {
            if let Err(error) = lock(a, b - a) {
                // Some OSes may lock a prefix before reporting failure. This
                // gap has no previous users, so rollback cannot unlock theirs.
                unlock(a, b - a);
                for &(a, b) in &gaps[..i] {
                    unlock(a, b - a);
                }
                self.coalesce();
                return Err(error);
            }
        }
        for s in &mut self.spans {
            if s.start >= start && s.end <= end {
                s.users += 1;
            }
        }
        self.spans.extend(gaps.into_iter().map(|(start, end)| Span {
            start,
            end,
            users: 1,
        }));
        self.spans.sort_by_key(|s| s.start);
        self.coalesce();
        Ok(())
    }
    fn release(&mut self, start: usize, end: usize, mut unlock: impl FnMut(usize, usize)) {
        self.split(start);
        self.split(end);
        for s in &mut self.spans {
            if s.start >= start && s.end <= end {
                s.users -= 1;
                if s.users == 0 {
                    unlock(s.start, s.end - s.start);
                }
            }
        }
        self.spans.retain(|s| s.users != 0);
        self.coalesce();
    }
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}
fn pages(ptr: usize, len: usize, page: usize) -> io::Result<(usize, usize)> {
    let end = ptr
        .checked_add(len)
        .and_then(|end| end.checked_add(page - 1))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pin range overflow"))?;
    Ok((ptr / page * page, end / page * page))
}
#[cfg(unix)]
fn page_size() -> io::Result<usize> {
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if value <= 0 {
        return Err(io::Error::other("cannot determine OS page size"));
    }
    Ok(value as usize)
}
#[cfg(not(unix))]
fn page_size() -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "RAM pinning requires Unix",
    ))
}
#[cfg(unix)]
fn lock(start: usize, len: usize) -> io::Result<()> {
    // SAFETY: the guard retains the borrowed regions; these page-rounded
    // addresses cover live allocations for the duration of the OS call.
    if unsafe { libc::mlock(start as *const libc::c_void, len) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
#[cfg(not(unix))]
fn lock(_: usize, _: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "RAM pinning requires Unix",
    ))
}
fn unlock(start: usize, len: usize) {
    #[cfg(unix)]
    if unsafe { libc::munlock(start as *const libc::c_void, len) } != 0 {
        log::warn!("cannot release RAM pin: {}", io::Error::last_os_error());
    }
    #[cfg(not(unix))]
    let _ = (start, len);
}

/// Successful page locks, released when this guard drops.
/// Borrowing prevents dropping or mutating the allocations while pinned.
#[derive(Default)]
#[must_use = "keep the guard alive for the intended RAM residency interval"]
pub struct PinGuard<'a> {
    regions: Vec<&'a [u8]>,
    ranges: Vec<(usize, usize)>,
    bytes: usize,
}
impl<'a> PinGuard<'a> {
    /// Page-rounded bytes held by this guard (shared pages counted once).
    pub fn pinned_bytes(&self) -> usize {
        self.bytes
    }

    /// Pin one immutable dataset's regions, atomically with respect to failure.
    ///
    /// # Errors
    /// Returns OS/resource errors or rejects an aggregate page count exceeding
    /// `budget`. Previously acquired regions remain locked after a failure.
    pub fn try_pin(&mut self, regions: &[&'a [u8]], budget: usize) -> io::Result<()> {
        let page = page_size()?;
        let mut ranges = self.ranges.clone();
        for bytes in regions.iter().filter(|r| !r.is_empty()) {
            ranges.push(pages(bytes.as_ptr().addr(), bytes.len(), page)?);
        }
        ranges.sort_unstable();
        let mut union: Vec<(usize, usize)> = Vec::new();
        for (a, b) in ranges {
            if let Some(last) = union.last_mut().filter(|last| a <= last.1) {
                last.1 = last.1.max(b);
            } else {
                union.push((a, b));
            }
        }
        let bytes = union
            .iter()
            .try_fold(0usize, |sum, (a, b)| sum.checked_add(b - a))
            .ok_or_else(|| io::Error::other("pin budget overflow"))?;
        if bytes > budget {
            return Err(io::Error::other("page-rounded pin budget exceeded"));
        }
        let mut registry = registry()
            .lock()
            .map_err(|_| io::Error::other("pin registry poisoned"))?;
        // Acquire the replacement union first. Existing references keep every
        // old range locked during rollback and during the subsequent release.
        for (i, &(a, b)) in union.iter().enumerate() {
            if let Err(error) = registry.acquire(a, b, lock, unlock) {
                for &(a, b) in &union[..i] {
                    registry.release(a, b, unlock);
                }
                return Err(error);
            }
        }
        for &(a, b) in &self.ranges {
            registry.release(a, b, unlock);
        }
        self.regions.extend(regions.iter().copied());
        self.ranges = union;
        self.bytes = bytes;
        Ok(())
    }
}
impl Drop for PinGuard<'_> {
    fn drop(&mut self) {
        // A poisoned registry still owns the locks and must release our refs.
        let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
        for &(a, b) in &self.ranges {
            registry.release(a, b, unlock);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_readers_keep_shared_pages_locked() {
        let mut r = Registry::default();
        let mut locks = Vec::new();
        let mut unlocks = Vec::new();
        r.acquire(
            0,
            8192,
            |a, n| {
                locks.push((a, n));
                Ok(())
            },
            |_, _| panic!(),
        )
        .unwrap();
        r.acquire(
            4096,
            12288,
            |a, n| {
                locks.push((a, n));
                Ok(())
            },
            |_, _| panic!(),
        )
        .unwrap();
        r.release(0, 8192, |a, n| unlocks.push((a, n)));
        assert_eq!(locks, [(0, 8192), (8192, 4096)]);
        assert_eq!(unlocks, [(0, 4096)]);
        assert!(r.spans.iter().all(|s| s.users == 1));
        r.release(4096, 12288, |a, n| unlocks.push((a, n)));
        assert!(r.spans.is_empty());
        assert_eq!(unlocks.iter().map(|v| v.1).sum::<usize>(), 12288);
    }
    #[test]
    fn failure_rolls_back_only_new_locks() {
        let mut r = Registry::default();
        r.acquire(4096, 8192, |_, _| Ok(()), |_, _| panic!())
            .unwrap();
        let mut released = Vec::new();
        assert!(r
            .acquire(
                0,
                12288,
                |a, _| if a == 8192 {
                    Err(io::Error::other("limit"))
                } else {
                    Ok(())
                },
                |a, n| released.push((a, n))
            )
            .is_err());
        assert_eq!(released, [(8192, 4096), (0, 4096)]);
        assert_eq!(
            r.spans,
            [Span {
                start: 4096,
                end: 8192,
                users: 1
            }]
        );
    }
    #[test]
    fn repeated_and_large_regions_keep_bounded_accounting() {
        let mut r = Registry::default();
        let end = usize::try_from(512u64 << 30).unwrap_or(1 << 30);
        r.acquire(0, end, |_, _| Ok(()), |_, _| panic!()).unwrap();
        // A second guard reuses the same lock, even for billion-vector buffers.
        r.acquire(0, end, |_, _| panic!("already locked"), |_, _| panic!())
            .unwrap();
        assert_eq!(r.spans.len(), 1);
        for i in 1..100 {
            r.acquire(i * 4096, (i + 1) * 4096, |_, _| panic!(), |_, _| panic!())
                .unwrap();
            r.release(i * 4096, (i + 1) * 4096, |_, _| panic!());
            assert_eq!(r.spans.len(), 1);
        }
        r.release(0, end, |_, _| panic!("another reader remains"));
        let mut released = Vec::new();
        r.release(0, end, |a, n| released.push((a, n)));
        assert_eq!(released, [(0, end)]);
        assert!(r.spans.is_empty());
    }
    #[test]
    fn page_rounding_and_overflow() {
        assert_eq!(pages(4097, 4096, 4096).unwrap(), (4096, 12288));
        assert!(pages(usize::MAX - 1, 10, 4096).is_err());
    }
    #[test]
    fn live_guards_keep_borrows_and_idempotent_page_counts() {
        let bytes = vec![1u8; 8192];
        let mut first = PinGuard::default();
        // Sandboxes and deployments may deny mlock. Its failure contract is
        // still checked; deterministic registry tests cover rollback separately.
        if first.try_pin(&[&bytes], usize::MAX).is_err() {
            assert_eq!(first.pinned_bytes(), 0);
            assert!(first.ranges.is_empty());
            return;
        }
        let total = first.pinned_bytes();
        assert!(total >= bytes.len());
        first.try_pin(&[&bytes], total).unwrap();
        assert_eq!(first.pinned_bytes(), total);
        let mut second = PinGuard::default();
        second.try_pin(&[&bytes], total).unwrap();
        drop(first);
        assert_eq!(second.pinned_bytes(), total);
        assert_eq!(bytes[0], 1);
    }
    #[test]
    fn bounded_guard_rejects_before_os_calls() {
        let bytes = [1u8; 17];
        let mut guard = PinGuard::default();
        assert!(guard.try_pin(&[&bytes], 0).is_err());
        assert_eq!(guard.pinned_bytes(), 0);
    }
}

impl std::fmt::Debug for PinGuard<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinGuard")
            .field("pinned_bytes", &self.bytes)
            .field("range_count", &self.ranges.len())
            .finish_non_exhaustive()
    }
}
