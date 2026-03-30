/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global allocator wrapper that tracks current/peak memory and enforces an optional cap.
///
/// When a cap is set via `set_limit()`, allocations that would exceed it return null,
/// which typically causes the Rust runtime to abort. This works on all platforms.
pub struct TrackingAllocator {
    current: AtomicUsize,
    peak: AtomicUsize,
    /// 0 means unlimited
    limit: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
        }
    }

    pub fn current_bytes(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    pub fn reset_peak(&self) {
        self.peak
            .store(self.current.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Set allocation cap in bytes. 0 = unlimited.
    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let limit = self.limit.load(Ordering::Relaxed);

        // Check cap before allocating
        if limit > 0 {
            let current = self.current.load(Ordering::Relaxed);
            if current + size > limit {
                return std::ptr::null_mut();
            }
        }

        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let current = self.current.fetch_add(size, Ordering::Relaxed) + size;
            self.peak.fetch_max(current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        self.current.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
