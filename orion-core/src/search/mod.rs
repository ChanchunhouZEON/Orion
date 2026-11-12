/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Search phases: query preparation → cascade sessions → graph traversal → output.
//!
//! `filtered` is an exact allowed-ID path; the runtime owns label lookup and
//! revision snapshots. `adaptive` supplies stopping policy, while `diagnostics`
//! observes either path without owning query state. SIMD/prefetch lives in vector.
#[cfg(feature = "benchmark")]
pub use diagnostics::metrics::*;
#[cfg(not(feature = "benchmark"))]
pub(crate) use diagnostics::metrics::*;

/// Maximum worker count accepted by the service.
pub const MAX_WORKERS: usize = 64;

/// Collect and clear the three measurements reported by the CLI.
/// Returns (f32 distance computations, byte distance computations, visits).
#[cfg(feature = "search-stats")]
pub fn take_search_counts() -> (u64, u64, u64) {
    (NDC_F32.drain(), NDC_I8.drain(), VISIT_COUNT.drain())
}

#[cfg(feature = "benchmark")]
static INCLUDE_EXTRAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Switch the extras ablation arm for a benchmark run.
#[cfg(feature = "benchmark")]
pub fn set_include_extras(value: bool) {
    INCLUDE_EXTRAS.store(value, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
pub(crate) fn include_extras() -> bool {
    #[cfg(feature = "benchmark")]
    {
        INCLUDE_EXTRAS.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "benchmark"))]
    {
        true
    }
}

// Input preparation and validation.
pub mod query;

// Stage composition and shared execution.
pub mod cascade;

// Neighbor expansion and candidate maintenance.
pub mod traversal;

// Original-metric scores and complete result batches.
pub(crate) mod output;

// Resolved internal-ID filtering; label metadata remains in the runtime.
mod filtered;

// Orthogonal stopping policy and observation.
pub mod adaptive;
pub mod diagnostics;
#[cfg(feature = "benchmark")]
pub use diagnostics::SearchProfile;
