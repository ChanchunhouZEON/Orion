/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Optional cheap rejection tier in the cascade.
///
/// The stage is a **factory** for per-query `PrefilterSession`s.
/// The session owns its per-query state internally (the encoded query
/// signature, etc.) and exposes the hot-loop methods. Trait shape is
/// object-safe so callers can dispatch via `&dyn PrefilterStage` —
/// e.g. from a CLI-driven runner where the concrete cascade isn't
/// known until parse-time.
///
/// Implementors:
/// * [`NoPrefilter`] — passthrough (e.g. SIFT/glove where the
///   admission tier itself is already cheap).
/// * [`jl::JlPrefilter`] — JL Sparse 1024-bit Hamming filter for
///   high-D L2 workloads (GIST).
/// # Safety
/// `shape()` must describe the base allocation borrowed by every session.
/// `open` must reject incompatible query shapes. A stage returning `None` must
/// handle arbitrary IDs without unchecked access to a finite base allocation.
///
pub unsafe trait PrefilterStage: Send + Sync {
    /// Logical dimension and row count, for checking recipe/base compatibility.
    fn shape(&self) -> Option<(usize, usize)> {
        None
    }
    /// Materialise a per-query session. Called once at search setup
    /// (not per hop). Implementations should keep this O(N) at most
    /// — avoid O(N²) dense rotations (use Hadamard if needed).
    ///
    /// The session's lifetime borrows from `&self`, so it stays valid
    /// for the duration of a single search call without needing
    /// 'static query data.
    fn open<'a>(&'a self, q: &[f32]) -> Box<dyn PrefilterSession + 'a>;
}

/// Per-query prefilter state. Holds the encoded query (signature,
/// padded form, etc.) and exposes the hot-loop methods. Created via
/// `PrefilterStage::open` once per search call; lives for the
/// duration of the search.
/// # Safety
/// Compaction may only remove input IDs, never invent or alter IDs. The output
/// length cannot exceed the input length and every retained slot must remain
/// initialized. Safe scalar distance methods must check their own ID bounds.
///
pub unsafe trait PrefilterSession: Send + Sync {
    /// In-place cmov-compact `id_scratch` to entries with distance
    /// strictly less than `threshold`.
    ///
    /// Implementations typically drive
    /// `orion_ann_vector::DistanceStream<DistanceKernelBackend>` with the lookahead-
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
    /// - Every ID must address a row in the session's base dataset.
    /// - The vector must be exclusively borrowed for the entire compaction.
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
