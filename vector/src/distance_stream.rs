/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! [`DistanceStream`] — queue-driven prefetch + 32-byte-chunk
//! compute + sink pipeline.
//!
//! ## Why a queue?
//!
//! The previous "prefetch_vertex(ids[i + lookahead])" form coupled
//! prefetch granularity to vertex granularity. That's only the
//! right shape for **single-line vertices** (one cache line per
//! vertex, e.g. SIFT u8 N=128 / glove100 i8 N=100). It under-issues
//! for multi-line vertices (i16 N=100 = 2-3 lines, GIST f32 = 30
//! lines) — the prefetch runway only covers the first line of each
//! upcoming vertex, leaving the trailing lines as demand-fetches
//! mid-compute.
//!
//! The queue-based shape decouples them: the caller pre-builds a
//! flat list of cache-line addresses **in the order they'll be
//! accessed**, and the stream walks that queue with a fixed-line
//! lookahead. Per-vertex the caller also records boundaries in
//! `vertex_starts` so the stream knows when to wrap one vertex's
//! accumulator + sink and start the next.
//!
//! ## Layout
//!
//! ```text
//! cache_lines:        [v0_l0, v0_l1, v0_l2, v1_l0, v2_l0, v2_l1, ...]
//!                   ▲────── vertex 0 ─────▲   ▲ v1 ▲──── vertex 2 ─────
//! vertex_starts:    0                       3       4
//! ids:             [id0,                   id1,    id2, ...]
//! ```
//!
//! ## Pipeline (steady state, per vertex)
//!
//! 1. For each cache line of this vertex, issue `prfm` on the line
//!    `lookahead_lines` ahead in `cache_lines`. Multi-line vertices
//!    issue multiple prefetches per outer iter; sub-/single-line
//!    vertices issue exactly one.
//! 2. Fold the vertex's `chunks_per_vert` × 32-byte windows into the
//!    SIMD accumulator (`K::step` is `#[inline(always)]`, accumulator
//!    stays in a register across chunks).
//! 3. Reduce + invoke `sink(id, dist)`. Sink runs while the next
//!    iter's prefetches are in DRAM, extending the effective
//!    compute window — the latency-hiding shape we want.

use crate::distance_fn::DistanceFn;
use crate::CACHE_LINE_BYTES;

/// Maximum far-prefetch requests per compute group. The historical value
/// 12 benefited GIST L2-KT with the old per-vertex scheduler; it remains
/// the initial budget, not a claim that the new policy is already tuned.
/// Set ORION_DSTREAM_SINK_BURST=0 to disable far prefetching.
#[inline]
fn sink_burst() -> usize {
    use std::sync::OnceLock;
    static BURST: OnceLock<usize> = OnceLock::new();
    *BURST.get_or_init(|| {
        std::env::var("ORION_DSTREAM_SINK_BURST")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 64)
            .unwrap_or(12)
    })
}

/// Grouping is configured once per process, in logical cache-line units.
/// A vector occupies ceil(compute_bytes / CACHE_LINE_BYTES) such units;
/// padding between vectors is not counted as compute work.
fn burst_group_config() -> (usize, usize) {
    use std::sync::OnceLock;
    static CONFIG: OnceLock<(usize, usize)> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let read = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| (1..=64).contains(&value))
                .unwrap_or(default)
        };
        (
            read("ORION_DSTREAM_BURST_TRIGGER_LINES", 4),
            read("ORION_DSTREAM_BURST_AHEAD_VECTORS", 4),
        )
    })
}

/// No accumulated burst debt: the next group can only fill its own window.
/// Far and near cursors deliberately remain separate, allowing a line first
/// fetched at a distance to be requested again by the near drip.
#[derive(Clone, Copy)]
struct BurstSchedule {
    vertices_per_group: usize,
    far_distance_lines: usize,
    budget: usize,
}

impl BurstSchedule {
    fn new(
        lines_per_vertex: usize,
        near_lines: usize,
        trigger_lines: usize,
        ahead_vectors: usize,
        budget: usize,
    ) -> Self {
        Self {
            vertices_per_group: trigger_lines.div_ceil(lines_per_vertex).max(1),
            far_distance_lines: near_lines
                .saturating_add(ahead_vectors.saturating_mul(lines_per_vertex)),
            budget,
        }
    }

    #[inline(always)]
    fn requests(self, next: &mut usize, consumed: usize, total: usize) -> std::ops::Range<usize> {
        // Skip already-consumed lines, and never resolve an index past the
        // real list end. `next` means next-to-issue, not last-issued.
        let start = (*next).max(consumed).min(total);
        let limit = consumed.saturating_add(self.far_distance_lines).min(total);
        let count = limit.saturating_sub(start).min(self.budget);
        *next = start + count;
        start..*next
    }
}

/// Iterator-shaped fetch + compute + sink pipeline. Generic over the
/// kernel `K` (any [`DistanceFn`] impl) and the vector dim `N`. The
/// streaming loop is monomorphised per (kernel, dim) so `K::step`
/// and `K::reduce` get fully inlined into the hot path; the sink
/// closure is also inlined so the OoO backend can reorder its
/// instructions across the next iter's prefetch issue.
///
/// ## Auxiliary slab prefetch
///
/// Some search paths read a parallel scalar slab in the sink — e.g.
/// the L2-kernel-trick path reads `‖x_i8‖²` from a per-vertex `i32`
/// array to reconstruct `‖q-x‖² = ‖q‖² + ‖x‖² - 2·IP`. Without help,
/// each sink call eats a cold random load. Set the auxiliary slab via
/// [`with_aux`](Self::with_aux) and `run()` will issue a parallel
/// `prfm` for `aux_ptr + ids[pf_front] × aux_elem_bytes` alongside the
/// existing base prefetch — turning that cold load into a warm one
/// without any new closure plumbing.
pub struct DistanceStream<'a, K: DistanceFn, const N: usize> {
    /// Pointer to the start of the (quantized) base buffer.
    base_ptr: *const u8,
    /// Per-vertex byte stride: `v_ptr = base_ptr + id × id_stride_bytes`.
    id_stride_bytes: usize,
    /// Per-vertex compute span in bytes (multiple of `K::CHUNK_BYTES`).
    compute_bytes: usize,
    /// Pointer to the (padded) query vector.
    query_ptr: *const u8,
    /// Vertex IDs to score.
    ids: &'a [u32],
    /// Lookahead in cache lines.
    lookahead_lines: usize,
    /// Optional auxiliary slab prefetched in parallel with the base.
    /// Null when unused; the run loop's null check is loop-invariant
    /// and gets hoisted into a flat 2-way dispatch by LLVM.
    aux_ptr: *const u8,
    /// Bytes per aux element (e.g. 4 for `i32` per-vertex norms).
    aux_elem_bytes: usize,
    _phantom: std::marker::PhantomData<&'a K>,
}

impl<'a, K: DistanceFn, const N: usize> DistanceStream<'a, K, N> {
    /// Construct. Prefetch addresses are computed inline in `run()`
    /// from `(base_ptr, ids[v], id_stride_bytes, line_offset)` —
    /// no pre-built queue needed.
    ///
    /// # Safety
    /// - `base_ptr` valid for `(id_max + 1) × id_stride_bytes` bytes.
    /// - `compute_bytes` must be a multiple of `K::CHUNK_BYTES`; zero-pad
    ///   both base rows and query to that boundary for shorter dimensions.
    /// - Each addressed base row must contain `compute_bytes` readable bytes.
    /// - `query_ptr` valid for `compute_bytes` bytes (padded).
    /// - `id_stride_bytes` should be a multiple of `CACHE_LINE_BYTES`
    ///   (128 B on M2) for clean prfm targeting; unaligned strides
    ///   work but waste one prfm per vert on the leading partial line.
    #[inline]
    pub unsafe fn new(
        base_ptr: *const u8,
        id_stride_bytes: usize,
        compute_bytes: usize,
        query_ptr: *const u8,
        ids: &'a [u32],
        lookahead_lines: usize,
    ) -> Self {
        debug_assert!(compute_bytes % K::CHUNK_BYTES == 0);
        Self {
            base_ptr,
            id_stride_bytes,
            compute_bytes,
            query_ptr,
            ids,
            lookahead_lines: lookahead_lines.max(1),
            aux_ptr: std::ptr::null(),
            aux_elem_bytes: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Attach an auxiliary slab to be prefetched alongside the base on
    /// each outer iter. Address resolved as
    /// `aux_ptr + ids[pf_front] × aux_elem_bytes`. Use for any
    /// per-vertex scalar that the sink reads — e.g. the L2 kernel
    /// trick's `‖x‖²` slab.
    ///
    /// # Safety
    ///
    /// `aux_ptr` must be a valid base for at least
    /// `(max(ids) + 1) × aux_elem_bytes` bytes. The aux prefetch is
    /// best-effort (`prfm pldl1strm`) — fetching a slightly out-of-
    /// range line is benign as long as the page is mapped.
    #[inline]
    pub unsafe fn with_aux(mut self, aux_ptr: *const u8, aux_elem_bytes: usize) -> Self {
        self.aux_ptr = aux_ptr;
        self.aux_elem_bytes = aux_elem_bytes;
        self
    }

    /// Drive the stream. **Not** `#[inline(always)]` — at each search
    /// call site we monomorphise per (K, N) and there are typically
    /// 2 such instantiations per query (i8 beam + f32 rerank). Forcing
    /// inline-always blows up icache; `#[inline]` lets LLVM share the
    /// monomorphic body across all call sites with the same (K, N).
    ///
    /// Compute is grouped by logical cache-line consumption. Low-dimensional
    /// vectors share one far burst; high-dimensional vectors can form a group
    /// by themselves. Each group issues a bounded burst before its final sink.
    /// The ordinary compute-time drip is retained, including repeat requests
    /// for lines previously touched by the independent far cursor.
    #[inline]
    pub fn run<Sink>(self, mut sink: Sink)
    where
        Sink: FnMut(u32, f32),
    {
        let n_verts = self.ids.len();
        if n_verts == 0 {
            return;
        }

        let pf_batch = self.lookahead_lines;
        // Lines per vertex — derived from `compute_bytes` rounded up
        // to the cache-line boundary. The dataset is **chunk-aligned**
        // (32-byte SIMD chunks), NOT cache-line-aligned, so for very
        // small vectors `stride` may be smaller than CACHE_LINE_BYTES
        // and several vertices share a single cache line in memory.
        // The prfm loop still issues one prfm per vertex (graph IDs
        // are random, so adjacent ids[] entries don't typically share
        // a line), but the address stride and the "vertex fits in one
        // line" predicate are derived from `stride` / `compute_bytes`.
        //
        // Production cases (`stride` = byte-stride between consecutive
        // vertices in memory):
        //   lpv = 1, stride = 32   → glove-25 i8  (vector spans ≤1 line,
        //                            **4 vectors per cache line in mem**)
        //   lpv = 1, stride = 128  → SIFT u8, glove-100 i8 (1 vec/line)
        //   lpv = 1, stride = 128  → glove-25 f32 (100B padded to 128)
        //   lpv = 2, stride = 256  → SIFT i16
        //   lpv = 4, stride = 512  → SIFT f32, glove-100 f32
        //   lpv = 30, stride = 3840→ GIST f32 (non-pow2 lpv)
        let lpv: usize = (self.compute_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
        let n_lines = n_verts * lpv;

        // Prologue: prfm first `min(pf_batch, n_lines)` cache lines —
        // base slab unconditionally, aux slab at v_idx-transitions.
        unsafe {
            let head = pf_batch.min(n_lines);
            for li in 0..head {
                Self::prfm_l1(self.resolve(li));
                if !self.aux_ptr.is_null() && self.aux_at_idx(li) {
                    Self::prfm_l1(self.aux_resolve(li));
                }
            }
        }

        let mut pf_front: usize = pf_batch.min(n_lines);
        let mut far_next = pf_batch.min(n_lines);
        let burst = sink_burst();
        let (trigger_lines, ahead_vectors) = burst_group_config();
        let schedule = BurstSchedule::new(lpv, pf_batch, trigger_lines, ahead_vectors, burst);
        // Disabled far prefetch uses one group, avoiding per-vector checks.
        let group_size = if burst == 0 {
            n_verts
        } else {
            schedule.vertices_per_group
        };

        let mut group_start = 0;
        while group_start < n_verts {
            let group_end = group_start.saturating_add(group_size).min(n_verts);
            let last = group_end - 1;
            // No per-vector threshold/window condition in this loop.
            for vi in group_start..last {
                sink(self.ids[vi], self.compute_vertex(vi, &mut pf_front));
            }

            let distance = self.compute_vertex(last, &mut pf_front);
            let consumed_lines = group_end * lpv;
            for line in schedule.requests(&mut far_next, consumed_lines, n_lines) {
                unsafe {
                    Self::prfm_l1(self.resolve(line));
                }
            }
            // Requests are issued before sink so its work contributes to the
            // available latency-hiding interval. This is not a residency guarantee.
            sink(self.ids[last], distance);
            group_start = group_end;
        }
    }

    // These helpers must inline into the stream loop. An outlined compute or
    // resolver call forces live SIMD accumulators through the stack and hides
    // the fixed dimension/stride from constant propagation.
    #[inline(always)]
    fn resolve(&self, idx: usize) -> *const u8 {
        let stride = self.id_stride_bytes;
        let base = self.base_ptr;
        let lpv = self.compute_bytes.div_ceil(CACHE_LINE_BYTES);
        let last_v = self.ids.len() - 1;
        const CACHE_LINE_LOG2: u32 = CACHE_LINE_BYTES.trailing_zeros();
        let stride_is_pow2 = stride.is_power_of_two();
        let stride_log2: u32 = if stride_is_pow2 {
            stride.trailing_zeros()
        } else {
            0
        };
        let lpv_is_pow2 = lpv.is_power_of_two();
        let lpv_log2: u32 = if lpv_is_pow2 { lpv.trailing_zeros() } else { 0 };
        let lpv_mask: usize = lpv.wrapping_sub(1);

        // SAFETY: v_clamped < n_verts, so ids[v_clamped] is in
        // bounds; the resulting addr is at most lpv*CACHE_LINE_BYTES
        // past the last vert's base, which is within the buffer
        // footprint assumed by the caller (compute_bytes is read
        // per vert).
        unsafe {
            if lpv == 1 && stride_is_pow2 {
                // Hot path A: low-dim packed (glove-25 i8 stride=32) or
                // 1-vec-per-line (SIFT u8 stride=128). id_off = id<<log2.
                let v_clamped = idx.min(last_v);
                let id = self.ids.as_ptr().add(v_clamped).read() as usize;
                base.add(id << stride_log2)
            } else if lpv_is_pow2 && stride_is_pow2 {
                // Hot path B: high-dim multi-line (SIFT f32, glove-100 f32).
                //   v_idx  = idx >> lpv_log2          (right shift)
                //   l_idx  = idx &  lpv_mask          (low bits)
                //   id_off = id  << stride_log2       (left shift)
                //   ln_off = l_idx << CACHE_LINE_LOG2 (left shift)
                let v_idx = idx >> lpv_log2;
                let l_idx = idx & lpv_mask;
                let v_clamped = v_idx.min(last_v);
                let id = self.ids.as_ptr().add(v_clamped).read() as usize;
                base.add((id << stride_log2) + (l_idx << CACHE_LINE_LOG2))
            } else if lpv == 1 {
                // Cold path C: lpv=1, stride non-pow2 (no current dataset).
                let v_clamped = idx.min(last_v);
                let id = self.ids.as_ptr().add(v_clamped).read() as usize;
                base.add(id * stride)
            } else {
                // Cold path D: GIST f32 lpv=30 / GIST u8 stride=960.
                let v_idx = idx / lpv;
                let l_idx = idx - v_idx * lpv;
                let v_clamped = v_idx.min(last_v);
                let id = self.ids.as_ptr().add(v_clamped).read() as usize;
                base.add(id * stride + (l_idx << CACHE_LINE_LOG2))
            }
        }
    }

    #[inline(always)]
    fn aux_resolve(&self, idx: usize) -> *const u8 {
        let lpv = self.compute_bytes.div_ceil(CACHE_LINE_BYTES);
        let lpv_is_pow2 = lpv.is_power_of_two();
        let lpv_log2 = lpv.trailing_zeros();
        let last_v = self.ids.len() - 1;
        let aux_base = self.aux_ptr;
        let aux_elem = self.aux_elem_bytes;

        unsafe {
            let v_idx = if lpv == 1 {
                idx
            } else if lpv_is_pow2 {
                idx >> lpv_log2
            } else {
                idx / lpv
            };
            let v_clamped = v_idx.min(last_v);
            let id = self.ids.as_ptr().add(v_clamped).read() as usize;
            aux_base.add(id * aux_elem)
        }
    }

    #[inline(always)]
    fn aux_at_idx(&self, idx: usize) -> bool {
        let lpv = self.compute_bytes.div_ceil(CACHE_LINE_BYTES);
        let lpv_is_pow2 = lpv.is_power_of_two();
        let lpv_mask = lpv.wrapping_sub(1);

        if lpv == 1 {
            true
        } else if lpv_is_pow2 {
            (idx & lpv_mask) == 0
        } else {
            idx % lpv == 0
        }
    }

    #[inline(always)]
    fn compute_vertex(&self, vi: usize, pf_front: &mut usize) -> f32 {
        let base = self.base_ptr;
        let stride = self.id_stride_bytes;
        let chunks_per_vert = self.compute_bytes / K::CHUNK_BYTES;
        let main_chunks_end = chunks_per_vert & !3;

        let mut acc = K::init();
        let mut acc1 = K::init();
        let mut acc2 = K::init();
        let mut acc3 = K::init();
        unsafe {
            let v_ptr = base.add(self.ids[vi] as usize * stride);
            let mut c = 0;
            while c < main_chunks_end {
                let off0 = c * K::CHUNK_BYTES;
                let off1 = (c + 1) * K::CHUNK_BYTES;
                let off2 = (c + 2) * K::CHUNK_BYTES;
                let off3 = (c + 3) * K::CHUNK_BYTES;

                K::step(
                    &mut acc,
                    v_ptr.add(off0) as *const K::Storage,
                    self.query_ptr.add(off0) as *const K::Storage,
                );
                K::step(
                    &mut acc1,
                    v_ptr.add(off1) as *const K::Storage,
                    self.query_ptr.add(off1) as *const K::Storage,
                );
                *pf_front += 1;
                Self::prfm_l1(self.resolve(*pf_front));
                if !self.aux_ptr.is_null() && self.aux_at_idx(*pf_front) {
                    Self::prfm_l1(self.aux_resolve(*pf_front));
                }
                K::step(
                    &mut acc2,
                    v_ptr.add(off2) as *const K::Storage,
                    self.query_ptr.add(off2) as *const K::Storage,
                );
                K::step(
                    &mut acc3,
                    v_ptr.add(off3) as *const K::Storage,
                    self.query_ptr.add(off3) as *const K::Storage,
                );
                #[cfg(target_arch = "x86_64")]
                {
                    *pf_front += 1;
                    Self::prfm_l1(self.resolve(*pf_front));
                    if !self.aux_ptr.is_null() && self.aux_at_idx(*pf_front) {
                        Self::prfm_l1(self.aux_resolve(*pf_front));
                    }
                }

                c += 4;
            }

            // ── Trailing-line prefetch ────────────────────────────
            // Per-iter drip above fires one prfm per `4 chunks`
            // = 128 B of compute = exactly one cache line. But
            // when `compute_bytes % CACHE_LINE_BYTES != 0`, the
            // tail loop below consumes a partial cache line that
            // the drip never covered — the per-vertex prfm count
            // is then `lpv - 1` instead of `lpv`, leaking one line
            // of runway per vertex.
            //
            // GIST L2-KT (stride=960 → 30 chunks → 28+2 split,
            // lpv=8 vs 7 drips) is the only production kernel that
            // trips this today; for kernels with
            // `chunks_per_vert % 4 == 0` this branch is provably
            // false and LLVM hoists it away.
            if chunks_per_vert > main_chunks_end {
                *pf_front += 1;
                Self::prfm_l1(self.resolve(*pf_front));
                if !self.aux_ptr.is_null() && self.aux_at_idx(*pf_front) {
                    Self::prfm_l1(self.aux_resolve(*pf_front));
                }
            }

            // On x86, consume a remaining pair with independent
            // accumulators before reducing the four-way loop. Native
            // SIFT128 uses two 64-byte chunks, so it takes this path
            // instead of serially updating one accumulator twice.
            // After the pair, advance the prefetch front once more.
            // Together with the trailing prefetch above, SIFT128
            // issues two regular prefetches for its two cache lines.
            #[cfg(target_arch = "x86_64")]
            if c + 1 < chunks_per_vert {
                let off0 = c * K::CHUNK_BYTES;
                let off1 = (c + 1) * K::CHUNK_BYTES;
                K::step(
                    &mut acc,
                    v_ptr.add(off0) as *const K::Storage,
                    self.query_ptr.add(off0) as *const K::Storage,
                );
                K::step(
                    &mut acc1,
                    v_ptr.add(off1) as *const K::Storage,
                    self.query_ptr.add(off1) as *const K::Storage,
                );
                c += 2;
                *pf_front += 1;
                Self::prfm_l1(self.resolve(*pf_front));
            }

            K::merge(&mut acc, acc1);
            K::merge(&mut acc2, acc3);
            K::merge(&mut acc, acc2);

            while c < chunks_per_vert {
                let off = c * K::CHUNK_BYTES;
                K::step(
                    &mut acc,
                    v_ptr.add(off) as *const K::Storage,
                    self.query_ptr.add(off) as *const K::Storage,
                );
                c += 1;
            }
        }

        K::reduce(acc)
    }

    /// `prfm pldl1strm` — pull line into L1d with **streaming** retention
    /// hint (line is expected to be referenced few times then evicted).
    /// On x86, maps to `_mm_prefetch(_MM_HINT_NTA)`.
    ///
    /// Why `strm` over `keep`: each prefetched cache line is read once
    /// (or `lpv` consecutive times for high-dim) per query and not
    /// referenced again — graph IDs are random, so we don't get
    /// re-hits across queries. `strm` lets the cache controller
    /// promote eviction into LRU more aggressively, freeing slots for
    /// the next query's working set instead of holding stale lines.
    ///
    /// `asm!` is volatile-by-default in Rust (no `pure`/`nomem`
    /// option set), so LLVM cannot elide the instruction or hoist
    /// it across other side-effecting code. `options(nostack,
    /// preserves_flags)` tells the optimizer the asm doesn't touch
    /// the stack pointer or condition flags, freeing it to schedule
    /// surrounding instructions more aggressively.
    #[inline(always)]
    unsafe fn prfm_l1(p: *const u8) {
        #[cfg(target_arch = "x86_64")]
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_NTA);
        #[cfg(target_arch = "aarch64")]
        std::arch::asm!(
            "prfm pldl1strm, [{x}]",
            x = in(reg) p,
            options(nostack, preserves_flags),
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = p;
        }
    }

    /// `prfm pldl2keep` — install line into L2 only, bypass L1d.
    /// On x86, maps to `_mm_prefetch(_MM_HINT_T1)`. Same volatile +
    /// scheduler-relaxed shape as `prfm_l1`.
    #[inline(always)]
    #[allow(dead_code)]
    unsafe fn prfm_l2(p: *const u8) {
        #[cfg(target_arch = "x86_64")]
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T1);
        #[cfg(target_arch = "aarch64")]
        std::arch::asm!(
            "prfm pldl2keep, [{x}]",
            x = in(reg) p,
            options(nostack, preserves_flags),
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = p;
        }
    }
}

#[cfg(test)]
mod burst_schedule_tests {
    use super::BurstSchedule;

    #[test]
    fn group_size_tracks_cache_line_work() {
        for (lines, expected) in [(1, 4), (2, 2), (4, 1), (8, 1), (30, 1)] {
            let schedule = BurstSchedule::new(lines, 10, 4, 4, 12);
            assert_eq!(schedule.vertices_per_group, expected);
            assert_eq!(schedule.far_distance_lines, 10 + 4 * lines);
        }
    }

    #[test]
    fn requests_stay_in_window_and_preserve_next_to_issue() {
        for lines in [1, 2, 8, 30] {
            for vertices in [1, 2, 3, 7, 33] {
                let schedule = BurstSchedule::new(lines, 10, 4, 4, 12);
                let total = lines * vertices;
                let mut next = 10.min(total);
                let mut end = 0;
                let mut issued = Vec::new();
                while end < vertices {
                    end = (end + schedule.vertices_per_group).min(vertices);
                    let consumed = end * lines;
                    let old_next = next;
                    let range = schedule.requests(&mut next, consumed, total);
                    assert_eq!(range.start, old_next.max(consumed).min(total));
                    assert!(range.len() <= 12);
                    assert!(range.end <= total.min(consumed + schedule.far_distance_lines));
                    issued.extend(range);
                }
                assert!(issued.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
        let schedule = BurstSchedule::new(1, 10, 4, 4, 12);
        let mut next = 10;
        assert_eq!(schedule.requests(&mut next, 4, 100).start, 10);
        assert!(schedule.requests(&mut next, 4, 100).is_empty());
        assert!(schedule.requests(&mut next, 100, 100).is_empty());
    }

    #[test]
    fn disabled_budget_emits_no_requests() {
        let schedule = BurstSchedule::new(8, 10, 4, 4, 0);
        let mut next = 10;
        assert!(schedule.requests(&mut next, 8, 100).is_empty());
    }
}
