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
pub(super) fn ns(value: Duration) -> u64 {
    value.as_nanos() as u64
}
pub(super) fn start(capacity: usize) {
    SAMPLES.with(|s| *s.borrow_mut() = Some(Vec::with_capacity(capacity.min(1_000_000))));
}
pub(super) fn finish() -> Vec<Sample> {
    SAMPLES.with(|s| s.borrow_mut().take().unwrap_or_default())
}
pub(super) fn record(sample: Sample) {
    SAMPLES.with(|s| {
        if let Some(samples) = s.borrow_mut().as_mut() {
            samples.push(sample);
        }
    });
}
