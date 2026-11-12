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
    /// Scoring representation, checked once before query sessions are opened.
    /// `None` denotes the no-op stage: it needs no encoded query or distance
    /// representation. Only `NoPrefilter` returns `None`.
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision>;
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
    /// Execute the complete rejection stage and report the work performed.
    /// The shared policy calls each kernel's existing distance and compaction
    /// methods. `NoPrefilterSession` overrides this entire operation with a
    /// constant outcome, without inspecting or modifying scratch.
    ///
    /// # Safety
    /// Every ID in the beam and candidate buffer must address this session's
    /// validated dataset. `search_list_size` must be nonzero.
    unsafe fn apply(
        &self,
        scratch: &mut crate::model::scratch::InMemSearchScratch,
        search_list_size: usize,
        lookahead_lines: usize,
    ) -> PrefilterOutcome {
        if scratch.pq.size() < search_list_size {
            return PrefilterOutcome::default();
        }
        let candidate_count = scratch.id_scratch.len();
        let frontier_size = scratch.pq.size();
        let worst_id = scratch.pq[frontier_size - 1].id;
        let mut threshold_evaluations = 0;
        if scratch.prefilter_observations == 0 || scratch.prefilter_last_worst_id != worst_id {
            threshold_evaluations = frontier_size;
            let mut frontier_distance_sum = 0.0f32;
            for i in 0..frontier_size {
                frontier_distance_sum += self.distance(scratch.pq[i].id);
            }
            scratch.prefilter_frontier_mean = frontier_distance_sum / (frontier_size as f32);
            scratch.prefilter_last_worst_id = worst_id;
        }
        // Preserve the original mean-of-frontier-means policy. Even when
        // the worst ID is unchanged, this hop contributes the cached mean.
        scratch.prefilter_frontier_mean_sum += scratch.prefilter_frontier_mean;
        scratch.prefilter_observations += 1;
        let cutoff = scratch.prefilter_frontier_mean_sum / (scratch.prefilter_observations as f32)
            * super::tuning::prefilter_slack();
        unsafe {
            self.filter_compact(&mut scratch.id_scratch, cutoff, lookahead_lines);
        }
        PrefilterOutcome {
            cutoff: Some(cutoff),
            threshold_evaluations,
            candidate_evaluations: candidate_count,
            rejected: candidate_count - scratch.id_scratch.len(),
        }
    }

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

/// Work performed by one filter step. Keeping reporting separate from the
/// session's object-safe operation lets production NoopObserver calls disappear.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PrefilterOutcome {
    cutoff: Option<f32>,
    threshold_evaluations: usize,
    candidate_evaluations: usize,
    rejected: usize,
}
impl PrefilterOutcome {
    #[inline(always)]
    pub(crate) fn observe<O: crate::search::diagnostics::SearchObserver>(
        self,
        observer: &mut O,
        beam_cutoff: Option<f32>,
    ) {
        if self.threshold_evaluations != 0 {
            observer.prefilter(self.threshold_evaluations, 0);
        }
        if self.cutoff.is_some() {
            observer.prefilter(0, self.candidate_evaluations);
        }
        observer.prefilter_outcome(
            self.cutoff,
            beam_cutoff,
            self.candidate_evaluations,
            self.rejected,
        );
    }
}
