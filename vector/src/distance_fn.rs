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

// ─── AVX-512 streaming impls (x86_64 + target_feature = "avx512f") ────
//
// Same trait surface as the NEON impls — `init / step / reduce / merge`
// — but at 64 bytes per chunk instead of 32, so each `step` does one
// full 512-bit-wide pass per side. CHUNK_BYTES = 64 here ripples
// through `DistanceStream` automatically via `K::CHUNK_BYTES`.
//
// Cfg: `cfg(all(target_arch = "x86_64", target_feature = "avx512f"))`.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for L2U8Distance {
    type Storage = u8;
    /// 16 i32 lanes — `_mm512_madd_epi16` produces these directly,
    /// no further widen needed (Σ at D=960 maxes ≈ 6.2 M, within i32).
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        // Split 64 bytes into two 32-byte halves, widen each to
        // 32× i16 via VPMOVZXBW (zero-extend u8 → i16). Difference
        // fits in i16 (range ±255). Squared via VPMADDWD → 16 i32
        // pair sums per half, two halves → 32 i32, summed into acc.
        let a_lo = _mm256_loadu_si256(base_chunk as *const __m256i);
        let a_hi = _mm256_loadu_si256(base_chunk.add(32) as *const __m256i);
        let q_lo = _mm256_loadu_si256(query_chunk as *const __m256i);
        let q_hi = _mm256_loadu_si256(query_chunk.add(32) as *const __m256i);

        let aw_lo = _mm512_cvtepu8_epi16(a_lo);
        let aw_hi = _mm512_cvtepu8_epi16(a_hi);
        let qw_lo = _mm512_cvtepu8_epi16(q_lo);
        let qw_hi = _mm512_cvtepu8_epi16(q_hi);

        let d_lo = _mm512_sub_epi16(aw_lo, qw_lo);
        let d_hi = _mm512_sub_epi16(aw_hi, qw_hi);

        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(d_lo, d_lo));
        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(d_hi, d_hi));
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_epi32(acc) as f32 }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi32(*into, src) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for L2U16Distance {
    type Storage = u16;
    /// 4 independent i64 chains (same shape as the NEON path) —
    /// VPMADDWD produces 16 i32 per half, widen to i64 via
    /// `_mm512_cvtepi32_epi64` (two halves × 8 lanes each).
    type Acc = (__m512i, __m512i, __m512i, __m512i);
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            )
        }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u16, query_chunk: *const u16) {
        // 64 B = 32 u16 = two 512-bit loads.
        let a_lo = _mm512_loadu_si512(base_chunk as *const __m512i);
        let a_hi = _mm512_loadu_si512(base_chunk.add(16) as *const __m512i);
        let q_lo = _mm512_loadu_si512(query_chunk as *const __m512i);
        let q_hi = _mm512_loadu_si512(query_chunk.add(16) as *const __m512i);

        // Per-lane abs-diff via signed sub on widened i32. For u16
        // |a-b| ≤ 65535 fits in i32 but the signed sub at i16 width
        // could underflow — we widen to i32 first to keep the math
        // straightforward.
        let aw_lo_lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_lo));
        let aw_lo_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_lo, 1));
        let aw_hi_lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_hi));
        let aw_hi_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_hi, 1));
        let qw_lo_lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(q_lo));
        let qw_lo_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(q_lo, 1));
        let qw_hi_lo = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(q_hi));
        let qw_hi_hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(q_hi, 1));

        let d0 = _mm512_sub_epi32(aw_lo_lo, qw_lo_lo);
        let d1 = _mm512_sub_epi32(aw_lo_hi, qw_lo_hi);
        let d2 = _mm512_sub_epi32(aw_hi_lo, qw_hi_lo);
        let d3 = _mm512_sub_epi32(aw_hi_hi, qw_hi_hi);

        // i32 * i32 → i64 element-wise. AVX-512 has `_mm512_mul_epi32`
        // which multiplies even lanes; we use `_mm512_mullo_epi32` to
        // produce 16 i32 lanes (low 32 of i64 product), then widen.
        // For our range (|d| ≤ 65535) the i32 product (≤ 2^32) fits in
        // i64 cleanly.
        let p0 = _mm512_mullo_epi32(d0, d0);
        let p1 = _mm512_mullo_epi32(d1, d1);
        let p2 = _mm512_mullo_epi32(d2, d2);
        let p3 = _mm512_mullo_epi32(d3, d3);

        // Widen i32 → i64 for accumulation, splitting each 16-lane
        // i32 vec into low + high halves (8 i64 each).
        macro_rules! widen_add {
            ($dst:expr, $p:expr) => {{
                let lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256($p));
                let hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64($p, 1));
                $dst = _mm512_add_epi64($dst, lo);
                $dst = _mm512_add_epi64($dst, hi);
            }};
        }
        widen_add!(acc.0, p0);
        widen_add!(acc.1, p1);
        widen_add!(acc.2, p2);
        widen_add!(acc.3, p3);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s01 = _mm512_add_epi64(acc.0, acc.1);
            let s23 = _mm512_add_epi64(acc.2, acc.3);
            _mm512_reduce_add_epi64(_mm512_add_epi64(s01, s23)) as f32
        }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = _mm512_add_epi64(into.0, src.0);
            into.1 = _mm512_add_epi64(into.1, src.1);
            into.2 = _mm512_add_epi64(into.2, src.2);
            into.3 = _mm512_add_epi64(into.3, src.3);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for JLHammingDistance {
    type Storage = u8;
    /// 8 i64 lanes — VPOPCNTQ output. Per-step max = 64 bytes ×
    /// 8 bits = 512, so even i32 would suffice, but i64 matches
    /// the natural VPOPCNTQ output width with no widening.
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const u8, query_chunk: *const u8) {
        let a = _mm512_loadu_si512(base_chunk as *const __m512i);
        let q = _mm512_loadu_si512(query_chunk as *const __m512i);
        let x = _mm512_xor_si512(a, q);
        // `_mm512_popcnt_epi64` requires `avx512vpopcntdq` (Ice Lake+,
        // Sapphire Rapids, Zen 4+). Without it the kernel won't
        // compile; the caller is expected to build with
        // `target-feature=+avx512vpopcntdq` per `.cargo/config.toml.example`.
        // Skylake-SP / Cascade Lake users should set
        // `cfg(not(target_feature = "avx512vpopcntdq"))` to route
        // back to the scalar fallback (TODO: bit-sliced popcount
        // fallback like the standalone `hamming_avx512_harley_seal`).
        let pc = _mm512_popcnt_epi64(x);
        *acc = _mm512_add_epi64(*acc, pc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_epi64(acc) as f32 }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi64(*into, src) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for IpI8Distance {
    type Storage = i8;
    /// 16 i32 lanes — VPMADDWD output. Σ at D=960 maxes ≈ 15.5 M,
    /// well within i32.
    type Acc = __m512i;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_si512() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i8, query_chunk: *const i8) {
        // 64 i8 → split into two 32-element halves, sign-extend to
        // i16 via VPMOVSXBW, multiply-add pairs via VPMADDWD.
        let a_lo = _mm256_loadu_si256(base_chunk as *const __m256i);
        let a_hi = _mm256_loadu_si256(base_chunk.add(32) as *const __m256i);
        let q_lo = _mm256_loadu_si256(query_chunk as *const __m256i);
        let q_hi = _mm256_loadu_si256(query_chunk.add(32) as *const __m256i);

        let aw_lo = _mm512_cvtepi8_epi16(a_lo);
        let aw_hi = _mm512_cvtepi8_epi16(a_hi);
        let qw_lo = _mm512_cvtepi8_epi16(q_lo);
        let qw_hi = _mm512_cvtepi8_epi16(q_hi);

        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(aw_lo, qw_lo));
        *acc = _mm512_add_epi32(*acc, _mm512_madd_epi16(aw_hi, qw_hi));
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -(_mm512_reduce_add_epi32(acc) as f32) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_epi32(*into, src) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for IpI16Distance {
    type Storage = i16;
    /// 4 independent i64 chains, same shape as NEON. VPMADDWD →
    /// 16 i32 lanes per pair, widen + accumulate into i64.
    type Acc = (__m512i, __m512i, __m512i, __m512i);
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe {
            (
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            )
        }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const i16, query_chunk: *const i16) {
        let a_lo = _mm512_loadu_si512(base_chunk as *const __m512i);
        let a_hi = _mm512_loadu_si512(base_chunk.add(16) as *const __m512i);
        let q_lo = _mm512_loadu_si512(query_chunk as *const __m512i);
        let q_hi = _mm512_loadu_si512(query_chunk.add(16) as *const __m512i);

        let m0 = _mm512_madd_epi16(a_lo, q_lo);
        let m1 = _mm512_madd_epi16(a_hi, q_hi);

        // Widen i32 → i64 and dispatch across 4 acc chains.
        let m0_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(m0));
        let m0_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(m0, 1));
        let m1_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(m1));
        let m1_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(m1, 1));

        acc.0 = _mm512_add_epi64(acc.0, m0_lo);
        acc.1 = _mm512_add_epi64(acc.1, m0_hi);
        acc.2 = _mm512_add_epi64(acc.2, m1_lo);
        acc.3 = _mm512_add_epi64(acc.3, m1_hi);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe {
            let s01 = _mm512_add_epi64(acc.0, acc.1);
            let s23 = _mm512_add_epi64(acc.2, acc.3);
            -(_mm512_reduce_add_epi64(_mm512_add_epi64(s01, s23)) as f32)
        }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe {
            into.0 = _mm512_add_epi64(into.0, src.0);
            into.1 = _mm512_add_epi64(into.1, src.1);
            into.2 = _mm512_add_epi64(into.2, src.2);
            into.3 = _mm512_add_epi64(into.3, src.3);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for IpF32Distance {
    type Storage = f32;
    /// Single 16-wide FMA accumulator — at 16 floats/chunk the per-
    /// step throughput is already 4 fmadds/cycle on Genoa, well-
    /// saturated by one chain.
    type Acc = __m512;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a = _mm512_loadu_ps(base_chunk);
        let q = _mm512_loadu_ps(query_chunk);
        *acc = _mm512_fmadd_ps(a, q, *acc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { -_mm512_reduce_add_ps(acc) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_ps(*into, src) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl DistanceFn for L2F32Distance {
    type Storage = f32;
    type Acc = __m512;
    const CHUNK_BYTES: usize = 64;

    #[inline(always)]
    fn init() -> Self::Acc {
        unsafe { _mm512_setzero_ps() }
    }
    #[inline(always)]
    unsafe fn step(acc: &mut Self::Acc, base_chunk: *const f32, query_chunk: *const f32) {
        let a = _mm512_loadu_ps(base_chunk);
        let q = _mm512_loadu_ps(query_chunk);
        let d = _mm512_sub_ps(a, q);
        *acc = _mm512_fmadd_ps(d, d, *acc);
    }
    #[inline(always)]
    fn reduce(acc: Self::Acc) -> f32 {
        unsafe { _mm512_reduce_add_ps(acc) }
    }
    #[inline(always)]
    fn merge(into: &mut Self::Acc, src: Self::Acc) {
        unsafe { *into = _mm512_add_ps(*into, src) }
    }
}

// ─── Scalar fallbacks for non-aarch64, non-AVX-512 targets ────────────

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx512f"))))]
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
