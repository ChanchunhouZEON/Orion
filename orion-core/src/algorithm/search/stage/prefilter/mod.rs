/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Prefilter stage
//!
//! Optional cheap rejection tier between graph-neighbour expansion
//! and PQ admission. Maintains a per-query running-mean threshold and
//! compacts the unseen-id list in place to only the candidates that
//! pass.
//!
//! ## Threshold update contract
//!
//! The threshold is **owned by the unified search loop**, not the
//! stage — the same PA-aligned shape we already had in
//! `search_l2_u8_q` (mean over the PQ's per-entry low-precision
//! distances × slack). The stage exposes:
//!
//! * `PrefilterSession::distance` for the threshold-mean recompute
//!   loop (scalar, called O(L) times when `pq_back_id` changes).
//! * `PrefilterSession::filter_compact` for the per-hop streaming
//!   compaction (drives `DistanceStream<DistanceKernelBackend>` with the
//!   prefetch + 4-way ILP shape we tuned earlier).
//!
//! ## Implementations
//!
//! * [`NoPrefilter`] (below) — zero-cost no-op.
//! * [`jl::JlPrefilter`] — JL Sparse 1024-bit Hamming filter (current
//!   `search_l2_u8_q` default).

pub mod jl;
pub mod jl_hadamard;
pub mod rabitq;

pub use jl::{JlMipsPrefilter, JlPrefilter};
pub use jl_hadamard::JlHadamardPrefilter;
pub use rabitq::RabitqPrefilter;

mod contract;
pub(crate) use contract::{PrefilterSession, PrefilterStage};

/// An official prefilter stage.
pub type Prefilter<'a> = dyn PrefilterStage + 'a;

/// Zero-cost no-op prefilter. Use at dispatch sites where the cascade
/// runs admission directly off the graph neighbours.
///
/// `Option::<&dyn PrefilterStage>::None` would also work, but
/// explicit `NoPrefilter` lets the dispatcher always have a stage
/// object to call `open()` on without a leading branch.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPrefilter;

/// Session for [`NoPrefilter`] — all methods are no-ops.
#[derive(Debug, Clone, Copy, Default)]
struct NoPrefilterSession;

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl PrefilterStage for NoPrefilter {
    #[inline(always)]
    fn open<'a>(&'a self, _q: &[f32]) -> Box<dyn PrefilterSession + 'a> {
        Box::new(NoPrefilterSession)
    }
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl PrefilterSession for NoPrefilterSession {
    #[inline(always)]
    unsafe fn filter_compact(
        &self,
        _id_scratch: &mut Vec<u32>,
        _threshold: f32,
        _lookahead_lines: usize,
    ) {
        // Search loops should branch on `Option<&dyn _>` and never
        // call this for the `None` case. Body kept as a fallback.
    }

    #[inline(always)]
    fn distance(&self, _vid: u32) -> f32 {
        0.0
    }
}
