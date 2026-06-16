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
//! * [`PrefilterStage::distance`] for the threshold-mean recompute
//!   loop (scalar, called O(L) times when `pq_back_id` changes).
//! * [`PrefilterStage::filter_compact`] for the per-hop streaming
//!   compaction (drives `DistanceStream<DistanceFn>` with the
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

/// Optional cheap rejection tier in the cascade.
///
/// The stage is a **factory** for per-query [`PrefilterSession`]s.
/// The session owns its per-query state internally (the encoded query
/// signature, etc.) and exposes the hot-loop methods. Trait shape is
/// object-safe so callers can dispatch via `&dyn PrefilterStage<N>` —
/// e.g. from a CLI-driven runner where the concrete cascade isn't
/// known until parse-time.
///
/// Implementors:
/// * [`NoPrefilter`] — passthrough (e.g. SIFT/glove where the
///   admission tier itself is already cheap).
/// * [`jl::JlPrefilter`] — JL Sparse 1024-bit Hamming filter for
///   high-D L2 workloads (GIST).
pub trait PrefilterStage<const N: usize>: Send + Sync {
    /// Materialise a per-query session. Called once at search setup
    /// (not per hop). Implementations should keep this O(N) at most
    /// — avoid O(N²) dense rotations (use Hadamard if needed).
    ///
    /// The session's lifetime borrows from `&self`, so it stays valid
    /// for the duration of a single search call without needing
    /// 'static query data.
    fn open<'a>(&'a self, q: &[f32; N]) -> Box<dyn PrefilterSession + 'a>;
}

/// Per-query prefilter state. Holds the encoded query (signature,
/// padded form, etc.) and exposes the hot-loop methods. Created via
/// [`PrefilterStage::open`] once per search call; lives for the
/// duration of the search.
pub trait PrefilterSession: Send + Sync {
    /// In-place cmov-compact `id_scratch` to entries with distance
    /// strictly less than `threshold`.
    ///
    /// Implementations typically drive
    /// `vector::DistanceStream<DistanceFn, N>` with the lookahead-
    /// prefetch shape. The closure pattern is:
    ///
    /// ```ignore
    /// .run(|id, dist| {
    ///     *id_out_ptr.add(w) = id;
    ///     w += (dist < threshold) as usize;
    /// });
    /// ```
    ///
    /// # Safety
    /// - `id_scratch` must be mutably aliasable for the in-place
    ///   compaction.
    /// - `lookahead_lines` should match `dstream_la_q()` for cache-
    ///   line prefetch alignment.
    unsafe fn filter_compact(
        &self,
        id_scratch: &mut Vec<u32>,
        threshold: f32,
        lookahead_lines: usize,
    );

    /// Scalar distance for the threshold-mean recompute loop. Called
    /// up to `search_list_size` times per recompute (~50× per query
    /// at GIST L=192 in steady state).
    fn distance(&self, vid: u32) -> f32;
}

/// Zero-cost no-op prefilter. Use at dispatch sites where the cascade
/// runs admission directly off the graph neighbours.
///
/// `Option::<&dyn PrefilterStage>::None` would also work, but
/// explicit `NoPrefilter` lets the dispatcher always have a stage
/// object to call `open()` on without a leading branch.
pub struct NoPrefilter;

/// Session for [`NoPrefilter`] — all methods are no-ops.
pub struct NoPrefilterSession;

impl<const N: usize> PrefilterStage<N> for NoPrefilter {
    #[inline(always)]
    fn open<'a>(&'a self, _q: &[f32; N]) -> Box<dyn PrefilterSession + 'a> {
        Box::new(NoPrefilterSession)
    }
}

impl PrefilterSession for NoPrefilterSession {
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
