/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! 32-byte-chunk distance kernels — the building blocks for
//! [`crate::DistanceStream`].
//!
//! ## SIMD design
//!
//! Each implementation is a tiny stateful unit:
//!
//!   * `Acc` is a **SIMD vector type** (`int32x4_t` / `uint32x4_t` /
//!     `int64x2_t`) — the accumulator stays in a NEON register across
//!     every chunk of a vertex.
//!   * `init()` returns a zeroed accumulator (NEON has no `Default`
//!     derive on `int32x4_t` etc.).
//!   * `step(acc, base_chunk, query_chunk)` folds one 32-byte window
//!     into `acc` using the **fewest, fastest** SIMD ops on M2:
//!     - **`IpI8Distance`** uses `sdot` (Apple Silicon's signed
//!       dot-product; one instruction does 16 i8×i8 multiplies +
//!       4-way partial sum, accumulated into `int32x4_t`). Two
//!       `sdot` ops cover a 32-byte chunk — **2 SIMD ops vs the
//!       8 of vmull+vpadal**, ~4× fewer instructions per chunk.
//!     - **`L2U8Distance`** uses `vabdq_u8 → vmull_u8 → vpadalq_u16`
//!       directly into the `uint32x4_t` acc (no intermediate
//!       horizontal reduce per chunk).
//!     - **`IpI16Distance`** uses `vmull_s16 → vpadalq_s32`
//!       directly into an `int64x2_t` acc.
//!   * `reduce(acc)` does the **single** horizontal reduce
//!     (`vaddvq_s32` / `vaddvq_u32` / `vaddvq_s64`) at the end of
//!     the vertex, applying the metric-specific sign / cast to f32.
//!
//! Why "32-byte chunks": NEON registers are 128-bit (16 B). One
//! `ldp q,q` load brings two adjacent NEON registers in a single
//! instruction = 32 B. The quantized buffer's 32-byte stride
//! alignment guarantees these are split-free on M2.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 32-byte-chunk streaming distance kernel.
///
/// Implementors don't see vertex-IDs or strides — only one 32-byte
/// window of base + query at a time. State lives entirely in `Acc`
/// (a SIMD vector type) which the caller carries across every chunk
/// of one vertex. The horizontal reduce happens **once** in
/// [`reduce`](DistanceFn::reduce) at the end of the vertex.
pub trait DistanceFn {
    /// Storage element type (`u8` / `i8` / `i16` / `f32`).
    type Storage: Copy;
    /// SIMD-vector accumulator. NEON intrinsic types are `Copy` but
    /// not `Default`; use [`init`](DistanceFn::init) to construct.
    type Acc: Copy;

    /// Bytes processed per [`step`](DistanceFn::step) call.
    /// Default 32 — one `ldp q,q` (two NEON registers, 16 + 16 B)
    /// per side. Override to 64 for kernels that benefit from
    /// 4-way ILP (= 4 independent accumulators); see
    /// [`IpF32Distance`] / [`L2F32Distance`]. The streaming caller
    /// loops `compute_bytes / CHUNK_BYTES` times per vertex.
    const CHUNK_BYTES: usize = 32;

    /// Zeroed accumulator at the start of each vertex.
    fn init() -> Self::Acc;

    /// Fold one [`CHUNK_BYTES`](DistanceFn::CHUNK_BYTES)-sized
    /// window into `acc`.
    ///
    /// # Safety
    /// `base_chunk` and `query_chunk` must each point to a
    /// `CHUNK_BYTES` region that's at least 16-byte aligned
    /// (`ldp q,q` boundary). The dataset's 32-byte stride alignment
    /// guarantees this for `id × STRIDE` offsets; stack `[T; N]`
    /// arrays at `T = u8/i8/i16/f32` get natural alignment.
    unsafe fn step(
        acc: &mut Self::Acc,
        base_chunk: *const Self::Storage,
        query_chunk: *const Self::Storage,
    );

    /// Collapse the SIMD accumulator into a final f32 distance
    /// ("smaller == closer"). Performs the single horizontal reduce
    /// + metric-specific sign + cast.
    fn reduce(acc: Self::Acc) -> f32;

    /// Elementwise add `src` into `into`. Used by the chunk-streaming
    /// `DistanceStream::run` path: each round computes UNROLL chunks
    /// into independent temporary accs, then folds them into the
    /// vertex-running acc here. Must be commutative and associative
    /// over the SIMD lanes (it always is for integer/float add).
    fn merge(into: &mut Self::Acc, src: Self::Acc);
}

// ─── L2 squared-Euclidean over u8 ──────────────────────────────────────

/// Squared-L2 over u8 storage. Acc is `uint32x4_t` — the four lanes
/// accumulate `Σ (a-b)²` partial sums and are summed once at
/// `reduce` time.
pub struct L2U8Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for L2U8Distance {
    type Storage = u8;
    type Acc = uint32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_u32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a_lo = vld1q_u8(base_chunk);
        let a_hi = vld1q_u8(base_chunk.add(16));
        let q_lo = vld1q_u8(query_chunk);
        let q_hi = vld1q_u8(query_chunk.add(16));

        let d_lo = vabdq_u8(a_lo, q_lo);
        let d_hi = vabdq_u8(a_hi, q_hi);

        // Squared diffs widened to u16x8, pairwise-summed into the
        // running u32x4 accumulator. No horizontal reduce here.
        let p_lo_l = vmull_u8(vget_low_u8(d_lo), vget_low_u8(d_lo));
        let p_lo_h = vmull_high_u8(d_lo, d_lo);
        let p_hi_l = vmull_u8(vget_low_u8(d_hi), vget_low_u8(d_hi));
        let p_hi_h = vmull_high_u8(d_hi, d_hi);

        *acc = vpadalq_u16(*acc, p_lo_l);
        *acc = vpadalq_u16(*acc, p_lo_h);
        *acc = vpadalq_u16(*acc, p_hi_l);
        *acc = vpadalq_u16(*acc, p_hi_h);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_u32(acc) as f32 }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_u32(*into, src) }
    }
}

// ─── L2 squared-Euclidean over u16 ─────────────────────────────────────

/// Squared-L2 over u16 storage. Used as the precise (per-dim ~256×
/// finer than u8) tier of the L2 cascade — PA's `quantize_bits 16`
/// pattern: u8 filter for cheap rejection, u16 distance for PQ
/// admission, f32 truth only at end-of-beam rerank.
///
/// ## Accumulator
///
/// `Acc = (u64x2, u64x2, u64x2, u64x2)` — 4 independent u64x2 chains
/// (same shape as [`IpI16Distance`]). Per-dim squared diff max is
/// `65535² ≈ 4.29 GB`; sum over D=960 maxes at `~4.12 TB`, which
/// overflows u32 (4.29 GB) by ~1000×. u64 is mandatory.
///
/// ## Chunk
///
/// 32 B per [`step`](DistanceFn::step) = 16 u16 lanes = two
/// `vld1q_u16` loads + `vabdq_u16` + four `vmull_u16` widens to u32 +
/// four `vpadalq_u32` widens to u64 across the four chains. Each
/// chain runs in parallel through M2's 4 NEON pipes, hiding the
/// `vpadal` latency.
pub struct L2U16Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for L2U16Distance {
    type Storage = u16;
    type Acc = (uint64x2_t, uint64x2_t, uint64x2_t, uint64x2_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0),
                vdupq_n_u64(0),
            )
        }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        // 32 bytes = 16 u16 lanes = two paired loads of u16x8.
        let a_lo = vld1q_u16(base_chunk);
        let a_hi = vld1q_u16(base_chunk.add(8));
        let q_lo = vld1q_u16(query_chunk);
        let q_hi = vld1q_u16(query_chunk.add(8));

        // `|a - b|` as u16 — `vabdq_u16` is one instruction.
        let d_lo = vabdq_u16(a_lo, q_lo);
        let d_hi = vabdq_u16(a_hi, q_hi);

        // Squared diffs: `vmull_u16` widens u16x4 → u32x4. Four u32x4
        // vectors per step (one per low/high half of d_lo and d_hi).
        let p_ll = vmull_u16(vget_low_u16(d_lo), vget_low_u16(d_lo));
        let p_lh = vmull_high_u16(d_lo, d_lo);
        let p_hl = vmull_u16(vget_low_u16(d_hi), vget_low_u16(d_hi));
        let p_hh = vmull_high_u16(d_hi, d_hi);

        // Pairwise-add widen u32x4 → u64x2 into 4 independent chains.
        // Splitting across chains breaks the `vpadalq_u32` dep chain so
        // the M2 OoO engine can issue all four in parallel.
        acc.0 = vpadalq_u32(acc.0, p_ll);
        acc.1 = vpadalq_u32(acc.1, p_lh);
        acc.2 = vpadalq_u32(acc.2, p_hl);
        acc.3 = vpadalq_u32(acc.3, p_hh);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s = vaddq_u64(vaddq_u64(acc.0, acc.1), vaddq_u64(acc.2, acc.3));
            // u64 → f32: at D=960 / u16 max range the value can reach
            // 4 TB which exceeds f32 mantissa (16.7 M). Distance
            // ordering is preserved by the monotone cast, so PQ
            // comparisons stay correct even at precision-loss scale.
            vaddvq_u64(s) as f32
        }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_u64(into.0, src.0);
            into.1 = vaddq_u64(into.1, src.1);
            into.2 = vaddq_u64(into.2, src.2);
            into.3 = vaddq_u64(into.3, src.3);
        }
    }
}

// ─── JL Sparse Hamming popcount ───────────────────────────────────────

/// Hamming distance over packed u8 signatures — the **cheapest tier**
/// of the L2 search cascade (PA's `quantize_mode=3`). Used as a
/// prefilter to drop candidates before the (much costlier) u8 PQ
/// admission distance.
///
/// ## Accumulator
///
/// `Acc = uint32x4_t` — same shape as [`L2U8Distance`]. The 4-way
/// `DistanceStream` unroll spawns 4 independent accs, each
/// accumulating up to `chunks_per_vert / 4` chunks. For the
/// production case (BITS=1024, STRIDE=128 → 4 chunks per vert), each
/// chain takes exactly **one** chunk, so per-chain max value is
/// `32 × 8 = 256` — comfortably within u32.
///
/// ## Chunk
///
/// 32 B per [`step`](DistanceFn::step) = two `vld1q_u8` loads per side
/// + `veorq_u8 → vcntq_u8` per half + two pairwise widens (u8 → u16,
/// u16 → u32) folded into the running u32x4 accumulator. The XOR-
/// popcount path is what makes Hamming ~24× cheaper per cmp than u8
/// squared-L2 at the same byte width (no widening multiplies).
pub struct JLHammingDistance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for JLHammingDistance {
    type Storage = u8;
    type Acc = uint32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_u32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a_lo = vld1q_u8(base_chunk);
        let a_hi = vld1q_u8(base_chunk.add(16));
        let q_lo = vld1q_u8(query_chunk);
        let q_hi = vld1q_u8(query_chunk.add(16));

        // Per-byte popcount of XOR — `vcntq_u8` is one instruction that
        // returns 16 bytes whose lanes each hold the popcount (0..=8)
        // of the corresponding input byte.
        let p_lo = vcntq_u8(veorq_u8(a_lo, q_lo));
        let p_hi = vcntq_u8(veorq_u8(a_hi, q_hi));

        // Pairwise-add widen u8 → u16 → u32, fold into running u32x4
        // acc. Each `vpaddlq_u8` pair-sums 16 u8 lanes into 8 u16
        // lanes; `vpaddlq_u16` then pair-sums 8 u16 lanes into 4 u32
        // lanes. The two halves accumulate independently (no inter-
        // half dep), keeping the M2 NEON pipes full.
        let h16_lo = vpaddlq_u8(p_lo);
        let h16_hi = vpaddlq_u8(p_hi);
        let h32_lo = vpaddlq_u16(h16_lo);
        let h32_hi = vpaddlq_u16(h16_hi);

        *acc = vaddq_u32(*acc, h32_lo);
        *acc = vaddq_u32(*acc, h32_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_u32(acc) as f32 }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_u32(*into, src) }
    }
}

// ─── MIPS-IP over i8 — uses `sdot` (Apple Silicon dot-product) ────────

/// Symmetric i8 inner product. Uses NEON `sdot` (one instruction =
/// 16 i8×i8 multiplies + 4-way partial sum into int32x4_t). 2 `sdot`
/// ops per 32-byte chunk vs 8 `vmull/vpadal` ops in the legacy path.
pub struct IpI8Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for IpI8Distance {
    type Storage = i8;
    type Acc = int32x4_t;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { vdupq_n_s32(0) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        // FEAT_DotProd `sdot` via inline asm (the std intrinsic
        // `vdotq_s32` is still nightly-gated on stable rustc 1.86).
        // Each `sdot` does 4× i8 multiply-add into i32 (16 i8 lanes
        // per call). 32-byte chunk = 2 sdot ops vs the legacy
        // 4×vmull + 4×vpadalq = 8 ops; dep-chain depth drops 4 → 2
        // on a single accumulator. Native on M1/M2/M3.
        let a_lo = vld1q_s8(base_chunk);
        let a_hi = vld1q_s8(base_chunk.add(16));
        let q_lo = vld1q_s8(query_chunk);
        let q_hi = vld1q_s8(query_chunk.add(16));
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) *acc,
            a   = in(vreg)    a_lo,
            b   = in(vreg)    q_lo,
            options(pure, nomem, nostack),
        );
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) *acc,
            a   = in(vreg)    a_hi,
            b   = in(vreg)    q_hi,
            options(pure, nomem, nostack),
        );
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -(vaddvq_s32(acc) as f32) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = vaddq_s32(*into, src) }
    }
}

// ─── MIPS-IP over i16 ─────────────────────────────────────────────────

/// Symmetric i16 inner product. No `sdot`-style instruction for i16
/// on M2, but we still keep the accumulator vector-resident
/// (`int64x2_t`, two lanes; i16 dot at N=100 maxes ≈ 100·32767² ≈
/// 10¹¹, comfortably within i64 across both lanes).
pub struct IpI16Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for IpI16Distance {
    type Storage = i16;
    /// 4 independent `int64x2_t` accumulators — same shape PA's
    /// well-tuned i16 kernel uses. Single-acc was the bottleneck:
    /// `vpadalq_s32` has ~4-cycle latency, four serial calls per
    /// chunk = 16 cycle critical path. With 4 acc chains the four
    /// `vpadalq_s32` calls in one step run in parallel through M2's
    /// 4 NEON pipes, dropping per-chunk latency to ~4 cycle.
    type Acc = (int64x2_t, int64x2_t, int64x2_t, int64x2_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                vdupq_n_s64(0),
                vdupq_n_s64(0),
                vdupq_n_s64(0),
                vdupq_n_s64(0),
            )
        }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        // 32 bytes = 16 i16 lanes = two paired loads of i16x8.
        let a_lo = vld1q_s16(base_chunk);
        let a_hi = vld1q_s16(base_chunk.add(8));
        let q_lo = vld1q_s16(query_chunk);
        let q_hi = vld1q_s16(query_chunk.add(8));

        // vmull_s16 widens 4 i16 lanes to 4 i32 products. Each goes
        // into its own independent acc chain — no serial dependency.
        let p_lo_l = vmull_s16(vget_low_s16(a_lo), vget_low_s16(q_lo));
        let p_lo_h = vmull_high_s16(a_lo, q_lo);
        let p_hi_l = vmull_s16(vget_low_s16(a_hi), vget_low_s16(q_hi));
        let p_hi_h = vmull_high_s16(a_hi, q_hi);

        acc.0 = vpadalq_s32(acc.0, p_lo_l);
        acc.1 = vpadalq_s32(acc.1, p_lo_h);
        acc.2 = vpadalq_s32(acc.2, p_hi_l);
        acc.3 = vpadalq_s32(acc.3, p_hi_h);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        // Tree reduce: 3 vaddq + 1 vaddvq + sign flip.
        unsafe {
            let s01 = vaddq_s64(acc.0, acc.1);
            let s23 = vaddq_s64(acc.2, acc.3);
            -(vaddvq_s64(vaddq_s64(s01, s23)) as f32)
        }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_s64(into.0, src.0);
            into.1 = vaddq_s64(into.1, src.1);
            into.2 = vaddq_s64(into.2, src.2);
            into.3 = vaddq_s64(into.3, src.3);
        }
    }
}

// ─── f32 inner product (Stage-2 truth, MIPS) ──────────────────────────

/// Symmetric f32 inner product. 32-byte chunk = 8 f32 lanes split into
/// two independent `float32x4_t` accumulators (lo, hi) — gives 2-way
/// ILP across chunks: lo's FMA chain doesn't depend on hi's, so the
/// OoO engine can issue both in parallel through M2's 4 NEON pipes.
pub struct IpF32Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for IpF32Distance {
    type Storage = f32;
    type Acc = (float32x4_t, float32x4_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { (vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a_lo = vld1q_f32(base_chunk);
        let a_hi = vld1q_f32(base_chunk.add(4));
        let q_lo = vld1q_f32(query_chunk);
        let q_hi = vld1q_f32(query_chunk.add(4));
        acc.0 = vfmaq_f32(acc.0, a_lo, q_lo);
        acc.1 = vfmaq_f32(acc.1, a_hi, q_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -vaddvq_f32(vaddq_f32(acc.0, acc.1)) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_f32(into.0, src.0);
            into.1 = vaddq_f32(into.1, src.1);
        }
    }
}

// ─── f32 squared L2 (Stage-2 truth, L2) ────────────────────────────────

/// Squared-L2 over f32 storage. 32-byte chunk = 8 f32 lanes split
/// into two independent accumulators for 2-way ILP.
pub struct L2F32Distance;

#[cfg(target_arch = "aarch64")]
impl DistanceFn for L2F32Distance {
    type Storage = f32;
    type Acc = (float32x4_t, float32x4_t);

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { (vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
    }

    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a_lo = vld1q_f32(base_chunk);
        let a_hi = vld1q_f32(base_chunk.add(4));
        let q_lo = vld1q_f32(query_chunk);
        let q_hi = vld1q_f32(query_chunk.add(4));
        let d_lo = vsubq_f32(a_lo, q_lo);
        let d_hi = vsubq_f32(a_hi, q_hi);
        acc.0 = vfmaq_f32(acc.0, d_lo, d_lo);
        acc.1 = vfmaq_f32(acc.1, d_hi, d_hi);
    }

    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { vaddvq_f32(vaddq_f32(acc.0, acc.1)) }
    }

    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = vaddq_f32(into.0, src.0);
            into.1 = vaddq_f32(into.1, src.1);
        }
    }
}

// ─── Scalar fallbacks for non-aarch64 ─────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for L2U8Distance {
    type Storage = u8;
    type Acc = u32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        for i in 0..32 {
            let a = *base_chunk.add(i) as i32;
            let b = *query_chunk.add(i) as i32;
            let d = (a - b) as u32;
            *acc = acc.wrapping_add(d * d);
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for L2U16Distance {
    type Storage = u16;
    type Acc = u64;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        for i in 0..16 {
            let a = *base_chunk.add(i) as i32;
            let b = *query_chunk.add(i) as i32;
            let d = (a - b).unsigned_abs() as u64;
            *acc = acc.wrapping_add(d * d);
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for JLHammingDistance {
    type Storage = u8;
    type Acc = u32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        for i in 0..32 {
            *acc = acc.wrapping_add(((*base_chunk.add(i)) ^ (*query_chunk.add(i))).count_ones());
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc as f32
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for IpI8Distance {
    type Storage = i8;
    type Acc = i32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        for i in 0..32 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i32) * (*query_chunk.add(i) as i32));
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for IpI16Distance {
    type Storage = i16;
    type Acc = i64;
    #[inline(always)]
    fn init() -> Self::Acc {
        0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        for i in 0..16 {
            *acc = acc.wrapping_add((*base_chunk.add(i) as i64) * (*query_chunk.add(i) as i64));
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -(acc as f32)
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into = into.wrapping_add(src);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for IpF32Distance {
    type Storage = f32;
    type Acc = f32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            *acc += *base_chunk.add(i) * *query_chunk.add(i);
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        -acc
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl DistanceFn for L2F32Distance {
    type Storage = f32;
    type Acc = f32;
    #[inline(always)]
    fn init() -> Self::Acc {
        0.0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        for i in 0..8 {
            let d = *base_chunk.add(i) - *query_chunk.add(i);
            *acc += d * d;
        }
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        acc
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        *into += src;
    }
}
