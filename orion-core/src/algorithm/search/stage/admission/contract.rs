/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::Neighbor;

/// PQ admission distance kernel.
///
/// Stage is a **factory** for per-query `AdmissionSession`s. The
/// session owns its per-query state (padded SIMD query, reconstruction
/// constants like `q_norm_sq`, etc.) and exposes the hot-loop methods.
/// Trait shape is object-safe so the runner can build the cascade via
/// `Box<dyn AdmissionStage>` without naming the concrete type.
/// # Safety
/// `shape()` must describe every session's base allocation. `open` must reject
/// incompatible query shapes before a session can access raw memory. A stage
/// returning `None` must support arbitrary graph IDs without unchecked access.
/// Sessions must remain valid for the lifetime of the stage borrow.
///
pub unsafe trait AdmissionStage: Send + Sync {
    /// Scoring representation, checked once before query sessions are opened.
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision>;
    /// Logical dimension and row count, for checking recipe/base compatibility.
    fn shape(&self) -> Option<(usize, usize)> {
        None
    }
    /// Materialise a per-query session. Called once per search;
    /// drives the quantization + any reconstruction-constant setup
    /// (e.g. `‖q‖²` for kernel-trick).
    fn open<'a>(&'a self, q: &[f32]) -> Box<dyn AdmissionSession + 'a>;
}

/// Per-query admission state. Created by `AdmissionStage::open`;
/// lives for the duration of a single search call.
/// # Safety
/// `admit_stream` must initialize every slot in the prefix it reports, return at most
/// the input length, and emit only IDs from the input. It must not retain `out`
/// or write outside the caller-provided slots. Safe methods must check their own
/// ID bounds; the unsafe streaming method may rely on its caller's bounds.
///
///
/// Implementing a session requires acknowledging its output initialization contract:
/// ```compile_fail
/// use orion_ann_core::algorithm::search::stage::AdmissionSession;
/// use orion_ann_core::model::Neighbor;
/// struct Unchecked;
/// impl AdmissionSession for Unchecked {
///     fn entry_distance(&self, _: u32) -> f32 { 0. }
///     unsafe fn admit_stream(&self, ids: &[u32], _: *mut Neighbor, _: f32, _: usize) -> usize {
///         ids.len() // no output was initialized
///     }
/// }
/// ```
pub unsafe trait AdmissionSession: Send + Sync {
    /// Single-vertex distance computed via the same kernel as
    /// [`admit_stream`](Self::admit_stream). Used at search start to
    /// insert the entry vertex into the PQ before the main loop
    /// begins.
    fn entry_distance(&self, vid: u32) -> f32;

    /// Stream distances over `id_scratch`, write surviving
    /// `(id, distance)` pairs into `out` via cmov-compact. Returns
    /// the number of survivors written.
    ///
    /// The unified search loop owns the staging-buffer offset; this
    /// method writes into the slice starting at `out`.
    ///
    /// # Safety
    /// - Every ID must address a row in the session's base dataset.
    /// - `out` must point to at least `id_scratch.len()` `Neighbor`
    ///   slots (the upper bound for cmov-compact admits).
    /// - `lookahead_lines` should match `dstream_la_q()`.
    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead_lines: usize,
    ) -> usize;
}
