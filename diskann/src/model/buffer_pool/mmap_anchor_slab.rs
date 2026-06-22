/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! mmap-backed anchor pair storage with lock-free per-slot atomic append.
//!
//! Uses a file-backed `mmap` (MAP_SHARED) so the OS page cache transparently
//! manages residency. Cold pages are evicted to disk automatically under
//! memory pressure — no userspace LRU/BPM needed.
//!
//! ## Layout
//!
//! Identical to `AnchorPairSlab`: per-anchor cache-line aligned slots with
//! `AtomicU32` count for lock-free concurrent append during Vamana build.
//!
//! ## Memory accounting
//!
//! mmap pages are NOT tracked by the Rust allocator, so heap peak stays low.
//! RSS is bounded by the OS page cache; unused pages are demand-paged (zero-fill).

use crate::common::ANNResult;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

const CACHE_LINE: usize = 64;
const SLOT_HEADER: usize = 8; // count (u32) + reserved (u32)

fn compute_slot_bytes(max_pairs: usize) -> usize {
    let raw = SLOT_HEADER + max_pairs * 8;
    (raw + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE
}

pub struct MmapAnchorSlab {
    mmap: memmap2::MmapMut,
    _file: File,
    path: PathBuf,
    num_anchors: usize,
    max_pairs: usize,
    slot_bytes: usize,
    /// Total pairs that were dropped due to capacity overflow.
    total_overflow: AtomicU32,
}

// Safety: mmap region is a contiguous byte buffer. Per-slot writes use
// AtomicU32 for the count field; pair data writes at reserved indices are
// non-overlapping across threads (each thread writes to its own reserved range).
unsafe impl Send for MmapAnchorSlab {}
unsafe impl Sync for MmapAnchorSlab {}

impl MmapAnchorSlab {
    /// Create a new mmap-backed slab. The backing file is created at `path`
    /// and truncated to the required size. Pages are demand-paged (zero-filled
    /// on first access).
    pub fn new(path: &Path, num_anchors: usize, max_pairs: usize) -> ANNResult<Self> {
        let slot_bytes = compute_slot_bytes(max_pairs);
        let total_bytes = num_anchors * slot_bytes;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_bytes as u64)?;

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            _file: file,
            path: path.to_path_buf(),
            num_anchors,
            max_pairs,
            slot_bytes,
            total_overflow: AtomicU32::new(0),
        })
    }

    #[inline]
    pub fn num_anchors(&self) -> usize {
        self.num_anchors
    }

    #[inline]
    pub fn max_pairs(&self) -> usize {
        self.max_pairs
    }

    #[inline]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    #[inline]
    fn slot_ptr(&self, anchor: usize) -> *mut u8 {
        unsafe { (self.mmap.as_ptr() as *mut u8).add(anchor * self.slot_bytes) }
    }

    // ── Lock-free atomic append ──────────────────────────────────────────────

    /// Atomically append pairs to an anchor slot. Multiple threads can
    /// concurrently append to *any* slot (even the same slot) — the AtomicU32
    /// count ensures non-overlapping index reservation.
    ///
    /// Atomically append arbitrary `(u32, u32)` pairs to a slot.
    pub fn atomic_append_pairs(&self, slot: usize, pairs: &[(u32, u32)]) -> usize {
        let ptr = self.slot_ptr(slot);
        let count_atomic = unsafe { &*(ptr as *const AtomicU32) };
        let base_idx = count_atomic.fetch_add(pairs.len() as u32, Ordering::Relaxed) as usize;
        let max = self.max_pairs;

        let mut written = 0;
        for (j, &pair) in pairs.iter().enumerate() {
            let idx = base_idx + j;
            if idx >= max {
                let over = pairs.len() - j;
                count_atomic.fetch_sub(over as u32, Ordering::Relaxed);
                self.total_overflow
                    .fetch_add(over as u32, Ordering::Relaxed);
                break;
            }
            let pair_offset = SLOT_HEADER + idx * 8;
            unsafe {
                let pair_ptr = ptr.add(pair_offset) as *mut (u32, u32);
                std::ptr::write(pair_ptr, pair);
            }
            written += 1;
        }
        written
    }

    /// Returns number of pairs actually written (capped at max_pairs).
    pub fn atomic_append(&self, anchor: usize, location: u32, pruned_ids: &[u32]) -> usize {
        let ptr = self.slot_ptr(anchor);
        let count_atomic = unsafe { &*(ptr as *const AtomicU32) };

        let base_idx = count_atomic.fetch_add(pruned_ids.len() as u32, Ordering::Relaxed) as usize;
        let max = self.max_pairs;

        let mut written = 0;
        for (j, &pid) in pruned_ids.iter().enumerate() {
            let idx = base_idx + j;
            if idx >= max {
                let over = pruned_ids.len() - j;
                count_atomic.fetch_sub(over as u32, Ordering::Relaxed);
                self.total_overflow
                    .fetch_add(over as u32, Ordering::Relaxed);
                break;
            }
            let pair_offset = SLOT_HEADER + idx * 8;
            unsafe {
                let pair_ptr = ptr.add(pair_offset) as *mut (u32, u32);
                std::ptr::write(pair_ptr, (location, pid));
            }
            written += 1;
        }
        written
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    /// Number of stored pairs for `anchor` (clamped to max_pairs).
    #[inline]
    pub fn count(&self, anchor: usize) -> usize {
        let ptr = self.slot_ptr(anchor);
        let count = unsafe { *(ptr as *const u32) } as usize;
        count.min(self.max_pairs)
    }

    /// Raw count for `anchor` (may exceed max_pairs if truncated).
    #[inline]
    pub fn raw_count(&self, anchor: usize) -> usize {
        let ptr = self.slot_ptr(anchor);
        (unsafe { *(ptr as *const u32) }) as usize
    }

    /// Compute truncation statistics: (num_at_capacity, total_overflow, max_count).
    pub fn truncation_stats(&self) -> (usize, u32, usize) {
        let mut at_cap = 0;
        let mut max_count = 0;
        for i in 0..self.num_anchors {
            let c = self.count(i);
            if c == self.max_pairs {
                at_cap += 1;
            }
            if c > max_count {
                max_count = c;
            }
        }
        (
            at_cap,
            self.total_overflow.load(Ordering::Relaxed),
            max_count,
        )
    }

    /// Get sorted pairs for an anchor as a slice.
    /// Only valid after `sort_all()`.
    #[inline]
    pub fn pairs(&self, anchor: usize) -> &[(u32, u32)] {
        let ptr = self.slot_ptr(anchor);
        let count = unsafe { *(ptr as *const u32) } as usize;
        let n = count.min(self.max_pairs);
        let pair_ptr = unsafe { ptr.add(SLOT_HEADER) as *const (u32, u32) };
        unsafe { std::slice::from_raw_parts(pair_ptr, n) }
    }

    /// Sort all anchor slots by (location, pruned_id) in parallel.
    pub fn sort_all(&mut self) {
        use rayon::prelude::*;
        let slot_bytes = self.slot_bytes;
        let max_pairs = self.max_pairs;
        let base = self.mmap.as_ptr() as usize;

        (0..self.num_anchors)
            .into_par_iter()
            .for_each(move |anchor| {
                let ptr = (base + anchor * slot_bytes) as *mut u8;
                let count = unsafe { *(ptr as *const u32) as usize }.min(max_pairs);
                if count > 1 {
                    let pair_ptr = unsafe { ptr.add(SLOT_HEADER) as *mut (u32, u32) };
                    let pairs = unsafe { std::slice::from_raw_parts_mut(pair_ptr, count) };
                    pairs.sort_unstable();
                }
            });
    }

    /// Advise OS for sequential read access (used before extract scan).
    pub fn advise_sequential(&self) {
        self.mmap.advise(memmap2::Advice::Sequential).ok();
    }

    /// Flush dirty pages to disk.
    pub fn flush(&self) -> ANNResult<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Total virtual size in bytes.
    pub fn virtual_bytes(&self) -> usize {
        self.mmap.len()
    }
}

impl Drop for MmapAnchorSlab {
    fn drop(&mut self) {
        // mmap is unmapped by memmap2::MmapMut::drop.
        // Remove the temp file.
        std::fs::remove_file(&self.path).ok();
    }
}

impl std::fmt::Debug for MmapAnchorSlab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_pairs: usize = (0..self.num_anchors).map(|a| self.count(a)).sum();
        f.debug_struct("MmapAnchorSlab")
            .field("num_anchors", &self.num_anchors)
            .field("max_pairs", &self.max_pairs)
            .field(
                "virtual_size",
                &format!("{:.1} MB", self.virtual_bytes() as f64 / 1_048_576.0),
            )
            .field("total_pairs", &total_pairs)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let dir = std::env::temp_dir().join("mmap_slab_test_basic");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.bin");

        let slab = MmapAnchorSlab::new(&path, 10, 8).unwrap();
        slab.atomic_append(0, 10, &[20, 30]);
        assert_eq!(slab.count(0), 2);
        let p = slab.pairs(0);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], (10, 20));
        assert_eq!(p[1], (10, 30));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sort() {
        let dir = std::env::temp_dir().join("mmap_slab_test_sort");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.bin");

        let mut slab = MmapAnchorSlab::new(&path, 2, 8).unwrap();
        slab.atomic_append(0, 10, &[2]);
        slab.atomic_append(0, 5, &[1]);
        slab.atomic_append(0, 10, &[1]);
        slab.sort_all();
        assert_eq!(slab.pairs(0), &[(5, 1), (10, 1), (10, 2)]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_overflow() {
        let dir = std::env::temp_dir().join("mmap_slab_test_overflow");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.bin");

        let slab = MmapAnchorSlab::new(&path, 1, 2).unwrap();
        slab.atomic_append(0, 1, &[1]);
        slab.atomic_append(0, 2, &[2]);
        slab.atomic_append(0, 3, &[3]); // overflow
        assert_eq!(slab.count(0), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_concurrent() {
        use std::sync::Arc;
        let dir = std::env::temp_dir().join("mmap_slab_test_conc");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.bin");

        let slab = Arc::new(MmapAnchorSlab::new(&path, 4, 64).unwrap());

        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let slab = slab.clone();
                std::thread::spawn(move || {
                    for i in 0..10u32 {
                        slab.atomic_append(1, tid, &[i]);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(slab.count(1), 40);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_file_cleanup() {
        let dir = std::env::temp_dir().join("mmap_slab_test_cleanup");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.bin");

        {
            let _slab = MmapAnchorSlab::new(&path, 10, 8).unwrap();
            assert!(path.exists());
        }
        // File should be removed on drop.
        assert!(!path.exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
