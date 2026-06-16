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

/// Sink-time burst-prefetch depth — number of additional `prfm` calls
/// issued in the sink callback per vertex, extending the prefetch
/// runway beyond what the per-iter drip can sustain.
///
/// Each vertex's sink (`sink(id, dist)`) does ~5-8 ns of compute (cmp
/// against cutoff + cmov-write into the staging buffer). That's a
/// free window where additional `prfm` instructions overlap the
/// sink's natural latency without competing with the main K::step
/// pipe. The burst targets cache lines **`pf_long_front`** ahead of
/// the main drip's `pf_front` — effectively running a second
/// prefetch tier at a longer lookahead distance.
///
/// **Default `12`** — chosen via burst sweep on GIST L2-KT:
///
/// | burst | L=48 QPS | L=192 QPS |
/// |-------|----------|-----------|
/// |   0   |  54,125  |  20,651   |  ← baseline
/// |   4   |  53,836  |  21,497   |  marginal — within DRAM latency cliff
/// |   8   |  55,175  |  21,150   |  ditto
/// |  12   |  65,762  |  25,384   |  ← **knee**: clears DRAM hiding threshold
/// |  16   |  64,374  |  23,010   |  plateau
/// |  32   |  66,270  |  25,298   |  plateau
///
/// At burst≤8 the long-range prefetches land just barely in time for
/// the next vertex — DRAM latency (~80 ns) isn't fully hidden by
/// per-vertex compute (~22 ns), so each demand-fault still stalls.
/// At burst≥12, the lookahead exceeds the ~3.6-vertex DRAM latency
/// wall and lines arrive ahead of need. Higher burst just queues
/// more in MSHR (no extra gain, no measurable loss).
///
/// +20-27% QPS on GIST L2-KT vs burst=0, recall bit-identical.
/// Override via `STAGED_DSTREAM_SINK_BURST`; set `0` to disable.
#[inline]
fn sink_burst() -> usize {
    use std::sync::OnceLock;
    static BURST: OnceLock<usize> = OnceLock::new();
    *BURST.get_or_init(|| {
        std::env::var("STAGED_DSTREAM_SINK_BURST")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v <= 64)
            .unwrap_or(12)
    })
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
    /// Per-vertex compute span in bytes (multiple of 32).
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

        // Auxiliary slab — captured by the closures below for the
        // parallel `prfm` walk. `aux_resolve` mirrors `resolve` but
        // only needs the v_idx (vertex index) — the aux array is
        // **one element per vertex** so the per-cache-line offset that
        // applies to multi-line base verts is dropped.
        let aux_base = self.aux_ptr;
        let aux_elem = self.aux_elem_bytes;
        let do_aux = !aux_base.is_null();
        let aux_resolve = |idx: usize| -> *const u8 {
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
        };
        // Only emit aux prfm at v_idx-transition lines so multi-line
        // verts don't issue duplicate prefetches for the same scalar
        // slot. For `lpv == 1` every line is a transition; for pow2
        // lpv it's a low-bit mask test; non-pow2 falls back to mod.
        let aux_at_idx = |idx: usize| -> bool {
            if lpv == 1 {
                true
            } else if lpv_is_pow2 {
                (idx & lpv_mask) == 0
            } else {
                idx % lpv == 0
            }
        };

        // Prologue: prfm first `min(pf_batch, n_lines)` cache lines —
        // base slab unconditionally, aux slab at v_idx-transitions.
        unsafe {
            let head = pf_batch.min(n_lines);
            for li in 0..head {
                Self::prfm_l1(resolve(li));
                if do_aux && aux_at_idx(li) {
                    Self::prfm_l1(aux_resolve(li));
                }
            }
        }

        let mut pf_front: usize = pf_batch.min(n_lines);
        // Long-range burst front — initialised at the same point as
        // `pf_front` but advanced ONLY by the sink-time burst, so it
        // races ahead per vertex while pf_front matches consumption.
        let mut pf_long_front: usize = pf_batch.min(n_lines);
        let burst = sink_burst();

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
                    if do_aux && aux_at_idx(pf_front) {
                        Self::prfm_l1(aux_resolve(pf_front));
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
                        pf_front += 1;
                        Self::prfm_l1(resolve(pf_front));
                        if do_aux && aux_at_idx(pf_front) {
                            Self::prfm_l1(aux_resolve(pf_front));
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
                    pf_front += 1;
                    Self::prfm_l1(resolve(pf_front));
                    if do_aux && aux_at_idx(pf_front) {
                        Self::prfm_l1(aux_resolve(pf_front));
                    }
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

            // ── Sink-time burst prefetch ────────────────────────────
            // Issue `burst` extra prfms targeting lines further ahead
            // than `pf_front`. Overlaps the sink callback's ~5-8 ns
            // of compute with prfm issuance, costing nothing in
            // wall-time. `pf_long_front` advances by `burst` per
            // vertex while `pf_front` only matches consumption, so
            // the long-range queue grows linearly with vertex count
            // — the asymptotic lookahead becomes `LA + burst × vi`.
            //
            // Branch on `burst > 0` so disabling via
            // `STAGED_DSTREAM_SINK_BURST=0` is a zero-cost no-op
            // (LLVM folds the OnceLock-derived constant after the
            // first call).
            if burst > 0 {
                unsafe {
                    for _ in 0..burst {
                        pf_long_front += 1;
                        if pf_long_front >= n_lines {
                            break;
                        }
                        Self::prfm_l1(resolve(pf_long_front));
                    }
                }
            }

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
