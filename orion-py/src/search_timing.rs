/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Opt-in clocks. Samples stay on the calling thread and are drained after
//! timed work; no per-query Python tuple/list construction is introduced.
use std::{cell::RefCell, time::Duration};
/// Input, GIL release, native search, GIL return, result wrapping, total FFI (ns).
pub(super) type Sample = (u64, u64, u64, u64, u64, u64);
thread_local! { static SAMPLES: RefCell<Option<Vec<Sample>>> = const { RefCell::new(None) }; }

/// Converts a measured interval to nanoseconds for the benchmark sample record.
pub(super) fn ns(value: Duration) -> u64 {
    value.as_nanos() as u64
}

/// Starts a fresh thread-local timing capture with bounded initial reservation.
pub(super) fn start(capacity: usize) {
    SAMPLES.with(|s| *s.borrow_mut() = Some(Vec::with_capacity(capacity.min(1_000_000))));
}

/// Takes the current thread's captured samples and disables capture until the next start.
pub(super) fn finish() -> Vec<Sample> {
    SAMPLES.with(|s| s.borrow_mut().take().unwrap_or_default())
}

/// Appends a probe sample only when timing capture is active on the current thread.
pub(super) fn record(sample: Sample) {
    SAMPLES.with(|s| {
        if let Some(samples) = s.borrow_mut().as_mut() {
            samples.push(sample);
        }
    });
}
