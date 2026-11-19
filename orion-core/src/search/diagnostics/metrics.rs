/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Optional per-worker search instrumentation. Different index pools can use
//! the same Rayon worker number, so the shared process totals must be atomic.
#[cfg(feature = "search-stats")]
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-local diagnostic counters accumulated outside the public result payload.
#[cfg(feature = "search-stats")]
#[repr(align(128))]
struct WorkerCounters([AtomicU64; COUNTERS]);

#[cfg(feature = "benchmark")]
const COUNTERS: usize = 10;
#[cfg(all(feature = "search-stats", not(feature = "benchmark")))]
const COUNTERS: usize = 3;

#[cfg(feature = "search-stats")]
static WORKERS: [WorkerCounters; crate::search::MAX_WORKERS] =
    [const { WorkerCounters([const { AtomicU64::new(0) }; COUNTERS]) }; crate::search::MAX_WORKERS];

/// Read-only handle to a process-wide experimental metric.
#[derive(Debug)]
pub struct Counter<const SLOT: usize>(());

impl<const SLOT: usize> Counter<SLOT> {
    /// Adds work to the current worker's diagnostic counter when search statistics are enabled.
    #[inline]
    pub(crate) fn add(&self, _count: u64) {
        #[cfg(feature = "search-stats")]
        if SLOT < COUNTERS {
            let worker = rayon::current_thread_index().unwrap_or(0) % WORKERS.len();
            WORKERS[worker].0[SLOT].fetch_add(_count, Ordering::Relaxed);
        }
    }

    /// Samples the aggregate worker counter without resetting it; concurrent updates may continue.
    #[cfg(feature = "benchmark")]
    pub fn sum(&self) -> u64 {
        WORKERS
            .iter()
            .map(|w| w.0[SLOT].load(Ordering::Relaxed))
            .sum()
    }

    /// Collects and clears worker counters, without promising one atomic snapshot across all workers.
    #[cfg(feature = "search-stats")]
    pub fn drain(&self) -> u64 {
        WORKERS
            .iter()
            .map(|w| w.0[SLOT].swap(0, Ordering::Relaxed))
            .sum()
    }

    /// Discards the current aggregate diagnostic count.
    #[cfg(feature = "benchmark")]
    pub fn reset(&self) {
        self.drain();
    }
}

pub const VISIT_COUNT: Counter<0> = Counter(());
pub const NDC_I8: Counter<1> = Counter(());
pub const NDC_F32: Counter<2> = Counter(());
pub const RAW_VISIT_COUNT: Counter<3> = Counter(());
pub const QUERY_COUNT: Counter<4> = Counter(());
pub const PRE_CONV_HOPS: Counter<5> = Counter(());
pub const POST_CONV_HOPS: Counter<6> = Counter(());
pub const PRE_CONV_ADMITS: Counter<7> = Counter(());
pub const POST_CONV_ADMITS: Counter<8> = Counter(());
#[cfg(feature = "benchmark")]
pub const SETUP_NS: Counter<9> = Counter(());

/// A zero-sized value outside experiments, so setup takes no clock readings.
pub(crate) struct SetupTimer {
    #[cfg(feature = "benchmark")]
    started: std::time::Instant,
}
impl SetupTimer {
    /// Starts a setup timer in benchmark builds and remains inert in production builds.
    #[inline]
    pub(crate) fn now() -> Self {
        Self {
            #[cfg(feature = "benchmark")]
            started: std::time::Instant::now(),
        }
    }

    /// Adds elapsed setup time to the benchmark counter when timing is enabled.
    #[inline]
    pub(crate) fn record(self) {
        #[cfg(feature = "benchmark")]
        SETUP_NS.add(self.started.elapsed().as_nanos() as u64);
    }
}
