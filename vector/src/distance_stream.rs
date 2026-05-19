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

/// Iterator-shaped fetch + compute + sink pipeline. Generic over the
/// kernel `K` (any [`DistanceFn`] impl) and the vector dim `N`. The
/// streaming loop is monomorphised per (kernel, dim) so `K::step`
/// and `K::reduce` get fully inlined into the hot path; the sink
/// closure is also inlined so the OoO backend can reorder its
/// instructions across the next iter's prefetch issue.
pub struct DistanceStream<'a, K: DistanceFn, const N: usize> {
    /// Pointer to the start of the (quantized) base buffer.
    base_ptr: *const u8,
    /// Per-vertex byte stride: `v_ptr = base_ptr + id × id_stride_bytes`.
    id_stride_bytes: usize,
    /// Per-vertex compute span in bytes (multiple of 32).
    compute_bytes: usize,
    /// Pointer to the (padded) query vector.
    query_ptr: *const u8,
    /// Vertex IDs to score.
    ids: &'a [u32],
    /// Lookahead in cache lines.
    lookahead_lines: usize,
    _phantom: std::marker::PhantomData<&'a K>,
}

impl<'a, K: DistanceFn, const N: usize> DistanceStream<'a, K, N> {
    /// Construct. Prefetch addresses are computed inline in `run()`
    /// from `(base_ptr, ids[v], id_stride_bytes, line_offset)` —
    /// no pre-built queue needed.
    ///
    /// # Safety
    /// - `base_ptr` valid for `(id_max + 1) × id_stride_bytes` bytes.
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
            _phantom: std::marker::PhantomData,
        }
    }

    /// Drive the stream. **Not** `#[inline(always)]` — at each search
    /// call site we monomorphise per (K, N) and there are typically
    /// 2 such instantiations per query (i8 beam + f32 rerank). Forcing
    /// inline-always blows up icache; `#[inline]` lets LLVM share the
    /// monomorphic body across all call sites with the same (K, N).
    ///
    /// **Pipeline shape:**
    ///
    /// 1. **Prologue** — prefetch the first `min(MSHR, stride_lines)`
    ///    cache lines, NOT a full `stride_lines` burst. Issuing more
    ///    than MSHR back-to-back oversaturates: only the first ~MSHR
    ///    fly, the rest queue serially behind in-flight slots. The
    ///    remaining runway gets filled by the per-iter drip during
    ///    the first few outer iterations.
    ///
    /// 2. **Per outer iter** (one vertex) — drip prfms until
    ///    `pf_front` is `stride_lines` ahead of the current vertex's
    ///    first cache line, capped at MSHR additional issues per
    ///    iter so we never burst more than the MSHR depth in one shot.
    ///    For single-line vertices this fires 0 or 1 prfms; for
    ///    multi-line vertices it fires up to `lines_per_vert`. The
    ///    position relationship is self-correcting: the steady-state
    ///    in-flight count tracks `stride_lines` cache lines.
    ///
    /// 3. Compute all `chunks_per_vert` chunks via 4-way unrolled
    ///    `K::step` into 4 register-resident accumulators, merge,
    ///    reduce, sink. The `chunks_per_vert % 4` tail handles the
    ///    rare case where compute_bytes isn't a multiple of 128 B
    ///    (current datasets always pad to a multiple of 128 B, so
    ///    the tail is dead code in production).
    #[inline]
    pub fn run<Sink>(self, mut sink: Sink)
    where
        Sink: FnMut(u32, f32),
    {
        let n_verts = self.ids.len();
        if n_verts == 0 {
            return;
        }

        let chunks_per_vert = self.compute_bytes / K::CHUNK_BYTES;
        let pf_batch = self.lookahead_lines;
        let stride = self.id_stride_bytes;
        let base = self.base_ptr;
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
        let last_v = n_verts.saturating_sub(1);

        // Two specialised resolve paths split by `lpv`. The dispatch
        // branch is a captured constant per query → predicted perfectly
        // and folded away by the inliner.
        //
        // Shift conventions used below:
        //   - `idx → v_idx` (only when lpv > 1): right shift
        //     `idx >> lpv_log2` to recover the vertex index.
        //   - `id → byte offset`: left shift `id << stride_log2`
        //     (replaces `id * stride` whenever stride is a power of 2;
        //     true on every prod dataset except GIST u8 with stride=960).
        //   - `l_idx → byte offset within vert`: left shift
        //     `l_idx << CACHE_LINE_LOG2` (CACHE_LINE_BYTES is pow2).
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

        // Flat 4-way dispatch — pull the inner `if stride_is_pow2`
        // up to the top level so LLVM produces 4 fully specialised
        // resolve paths in BOTH the prologue and the main-loop drip.
        // The previous nested shape kept the inner branch as a `csel`
        // in the main-loop drip (LBB310_20 emitted both `lsl` and
        // `mul` on every prfm and selected — wasted ~1 cycle/prfm).
        // Flattening forces a clean `b.eq` jump at iter top so each
        // iter computes only the address arithmetic for its case.
        let resolve = |idx: usize| -> *const u8 {
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
        };

        // Prologue: prfm first `min(pf_batch, n_lines)` cache lines.
        unsafe {
            let head = pf_batch.min(n_lines);
            for li in 0..head {
                Self::prfm_l1(resolve(li));
            }
        }

        let mut pf_front: usize = pf_batch.min(n_lines);

        let main_chunks_end = chunks_per_vert & !3;
        let mut vi = 0usize;
        while vi < n_verts {
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
                    pf_front += 1;
                    Self::prfm_l1(resolve(pf_front));
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
                        pf_front += 1;
                        Self::prfm_l1(resolve(pf_front));
                    }

                    c += 4;
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

            sink(self.ids[vi], K::reduce(acc));
            vi += 1;
        }
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
