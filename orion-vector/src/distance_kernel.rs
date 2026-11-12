/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Chunked distance kernels — the building blocks for
//! [`crate::DistanceStream`].
//!
//! Built-in streams select a CPU backend once before their loop. The public
//! `Acc` and `CHUNK_BYTES` types remain compile-time contracts for callers that
//! use the primitive methods directly; runtime dispatch substitutes a private
//! kernel only for a complete stream. Its accumulator never crosses an ISA
//! boundary. Backends use 32- or 64-byte chunks and preserve the row layout.
//!
//! ## AArch64 SIMD design
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
//!     - **`IpI8Distance`** uses `sdot` when dot product is available (Apple Silicon's signed
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

/// Chunked streaming distance kernel.
///
/// Implementors don't see vertex-IDs or strides — only one `CHUNK_BYTES`
/// window of base + query at a time. State lives entirely in `Acc`
/// (a SIMD vector type) which the caller carries across every chunk
/// of one vertex. The horizontal reduce happens **once** in
/// [`reduce`](DistanceKernelBackend::reduce) at the end of the vertex.
pub trait DistanceKernelBackend {
    /// Execute a stream. Custom kernels use the same statically inlined loop;
    /// built-in kernels override this hook to select a CPU backend once.
    #[inline(always)]
    fn run_stream<S: FnMut(u32, f32)>(stream: crate::DistanceStream<'_, Self>, sink: S)
    where
        Self: Sized,
    {
        stream.run_static(sink);
    }

    /// Storage element type (`u8` / `i8` / `i16` / `f32`).
    type Storage: Copy;
    /// SIMD-vector accumulator. NEON intrinsic types are `Copy` but
    /// not `Default`; use [`init`](DistanceKernelBackend::init) to construct.
    type Acc: Copy;

    /// Bytes processed per [`step`](DistanceKernelBackend::step) call.
    /// Default 32 — one `ldp q,q` (two NEON registers, 16 + 16 B)
    /// per side. Override to 64 for kernels that benefit from
    /// 4-way ILP (= 4 independent accumulators); see
    /// [`IpF32Distance`] / [`L2F32Distance`]. The streaming caller
    /// loops `compute_bytes / CHUNK_BYTES` times per vertex.
    const CHUNK_BYTES: usize = 32;

    /// Block boundary used to pad a row for [`crate::DistanceStream`].
    /// Built-in x86 kernels accept a single 32-byte row through AVX2 even
    /// when their primitive `step` uses 64 bytes. Larger rows keep the
    /// compile-time block width. Direct `step` callers must still provide
    /// `CHUNK_BYTES` bytes; this hook only describes whole-stream dispatch.
    #[inline(always)]
    fn stream_block_bytes(_row_bytes: usize) -> usize {
        Self::CHUNK_BYTES
    }

    /// Query bytes consumed per base chunk. Ordinary metrics advance both
    /// pointers equally; ADC consumes one 256-entry f32 table per code byte.
    const QUERY_CHUNK_BYTES: usize = Self::CHUNK_BYTES;

    /// Conservative coordinate limit for exact integer accumulation. Runtime
    /// padded APIs reject larger dimensions instead of silently wrapping.
    const MAX_COORDINATES: usize = usize::MAX;

    /// Zeroed accumulator at the start of each vertex.
    fn init() -> Self::Acc;

    /// Fold one [`CHUNK_BYTES`](DistanceKernelBackend::CHUNK_BYTES)-sized
    /// window into `acc`.
    ///
    /// # Safety
    /// `base_chunk` must contain `CHUNK_BYTES` readable bytes; `query_chunk`
    /// must contain `QUERY_CHUNK_BYTES` readable bytes. Both must satisfy the
    /// implementation's alignment requirements (natural `Storage` alignment
    /// for ordinary metrics; f32 alignment for the ADC query table). SIMD
    /// alignment is not required. Padding participates
    /// in the computation and must be zero when representing shorter vectors.
    /// Accumulated integer coordinates must not exceed `MAX_COORDINATES`.
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
    /// over the SIMD lanes for exact integer arithmetic. Floating-point kernels
    /// may differ in rounding when their reduction order changes.
    fn merge(into: &mut Self::Acc, src: Self::Acc);
}

// Public metric markers keep their existing compile-time chunk/accumulator API.
// Stream execution selects a private backend without exposing its SIMD types.

/// Squared-L2 over u8 storage. On NEON, Acc is `uint32x4_t` — the four lanes
/// accumulate `Σ (a-b)²` partial sums and are summed once at
/// `reduce` time.
pub struct L2U8Distance;

/// Negative unsigned-byte inner product, with widened u64 accumulation.
pub struct IpU8Distance;

/// Squared-L2 over u16 storage. Used as the precise (per-dim ~256×
/// finer than u8) tier of the L2 cascade — PA's `quantize_bits 16`
/// pattern: u8 filter for cheap rejection, u16 distance for PQ
/// admission, f32 truth only at end-of-beam rerank.
///
/// ## Accumulator
///
/// On NEON, `Acc = (u64x2, u64x2, u64x2, u64x2)` — 4 independent u64x2 chains
/// (same shape as [`IpI16Distance`]). Per-dim squared diff max is
/// `65535² ≈ 4.29 GB`; sum over D=960 maxes at `~4.12 TB`, which
/// overflows u32 (4.29 GB) by ~1000×. u64 is mandatory.
///
/// ## Chunk
///
/// 32 B per [`step`](DistanceKernelBackend::step) = 16 u16 lanes = two
/// `vld1q_u16` loads + `vabdq_u16` + four `vmull_u16` widens to u32 +
/// four `vpadalq_u32` widens to u64 across the four chains. Each
/// chain runs in parallel through M2's 4 NEON pipes, hiding the
/// `vpadal` latency.
pub struct L2U16Distance;

/// Hamming distance over packed u8 signatures — the **cheapest tier**
/// of the L2 search cascade (PA's `quantize_mode=3`). Used as a
/// prefilter to drop candidates before the (much costlier) u8 PQ
/// admission distance.
///
/// ## Accumulator
///
/// On NEON, `Acc = uint32x4_t` — same shape as [`L2U8Distance`]. The 4-way
/// `DistanceStream` unroll spawns 4 independent accs, each
/// accumulating up to `chunks_per_vert / 4` chunks. For the
/// production case (BITS=1024, STRIDE=128 → 4 chunks per vert), each
/// chain takes exactly **one** chunk, so per-chain max value is
/// `32 × 8 = 256` — comfortably within u32.
///
/// ## Chunk
///
/// 32 B per [`step`](DistanceKernelBackend::step) = two `vld1q_u8` loads per side
/// plus `veorq_u8 → vcntq_u8` per half + two pairwise widens (u8 → u16,
/// u16 → u32) folded into the running u32x4 accumulator. The XOR-
/// popcount path is what makes Hamming ~24× cheaper per cmp than u8
/// squared-L2 at the same byte width (no widening multiplies).
pub struct JLHammingDistance;

/// Symmetric i8 inner product. With dot product support, uses NEON `sdot` (one instruction =
/// 16 i8×i8 multiplies + 4-way partial sum into int32x4_t). 2 `sdot`
/// ops per 32-byte chunk vs 8 `vmull/vpadal` ops in the baseline NEON path.
pub struct IpI8Distance;

/// Symmetric i16 inner product. No `sdot`-style instruction for i16
/// on M2, but we still keep the accumulator vector-resident
/// (`int64x2_t`, two lanes; i16 dot at N=100 maxes ≈ 100·32767² ≈
/// 10¹¹, comfortably within i64 across both lanes).
pub struct IpI16Distance;

/// Symmetric f32 inner product. 32-byte chunk = 8 f32 lanes split into
/// two independent `float32x4_t` accumulators (lo, hi) — gives 2-way
/// ILP across chunks: lo's FMA chain doesn't depend on hi's, so the
/// OoO engine can issue both in parallel through M2's 4 NEON pipes.
pub struct IpF32Distance;

/// Squared-L2 over f32 storage. 32-byte chunk = 8 f32 lanes split
/// into two independent accumulators for 2-way ILP.
pub struct L2F32Distance;

#[cfg(all(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512dq",
        target_feature = "avx512vl",
        target_feature = "fma"
    ),
    not(target_feature = "avx512vpopcntdq")
))]
use crate::stream_kernels::avx512 as selected;
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
use crate::stream_kernels::dotprod as selected;
#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
use crate::stream_kernels::neon as selected;
#[cfg(all(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512dq",
        target_feature = "avx512vl",
        target_feature = "fma"
    ),
    target_feature = "avx512vpopcntdq"
))]
use crate::stream_kernels::popcnt as selected;
#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512dq",
        target_feature = "avx512vl",
        target_feature = "fma"
    )
)))]
use crate::stream_kernels::scalar as selected;

macro_rules! public_kernel {
    ($name:ident, $dispatch:ident) => {
        public_kernel!($name, $dispatch, true, true);
    };
    ($name:ident, $dispatch:ident, $arm_ready:expr, $x86_ready:expr) => {
        impl DistanceKernelBackend for $name {
            type Storage = <selected::$name as DistanceKernelBackend>::Storage;
            type Acc = <selected::$name as DistanceKernelBackend>::Acc;
            const CHUNK_BYTES: usize = <selected::$name as DistanceKernelBackend>::CHUNK_BYTES;
            const MAX_COORDINATES: usize =
                <selected::$name as DistanceKernelBackend>::MAX_COORDINATES;
            #[inline(always)]
            fn stream_block_bytes(row_bytes: usize) -> usize {
                #[cfg(target_arch = "x86_64")]
                if row_bytes <= 32 {
                    return 32;
                }
                let _ = row_bytes;
                Self::CHUNK_BYTES
            }
            #[inline(always)]
            fn init() -> Self::Acc {
                selected::$name::init()
            }
            #[inline(always)]
            unsafe fn step(
                acc: &mut Self::Acc,
                base: *const Self::Storage,
                query: *const Self::Storage,
            ) {
                unsafe { selected::$name::step(acc, base, query) }
            }
            #[inline(always)]
            fn reduce(acc: Self::Acc) -> f32 {
                selected::$name::reduce(acc)
            }
            #[inline(always)]
            fn merge(into: &mut Self::Acc, src: Self::Acc) {
                selected::$name::merge(into, src)
            }
            #[inline(always)]
            fn run_stream<S: FnMut(u32, f32)>(stream: crate::DistanceStream<'_, Self>, sink: S) {
                // Explicit ISA builds retain the original inlining boundary.
                // Baseline ARM already guarantees NEON; only optional DotProd
                // needs runtime selection. Portable x86 uses the dispatcher.
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                if $arm_ready {
                    stream.run_static(sink);
                    return;
                }
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "avx512dq",
                    target_feature = "avx512vl",
                    target_feature = "fma"
                ))]
                if $x86_ready {
                    if stream.compute_bytes() == 32 {
                        // These compile-time AVX-512 features also guarantee
                        // AVX2 + FMA. Select once, before the streaming loop;
                        // the wider static path below remains unchanged.
                        unsafe {
                            crate::stream_kernels::$dispatch(
                                stream,
                                sink,
                                crate::stream_kernels::DistanceIsa::Avx2,
                            );
                        }
                        return;
                    }
                    stream.run_static(sink);
                    return;
                }
                // SAFETY: the cached selection checks every required feature.
                unsafe {
                    crate::stream_kernels::$dispatch(
                        stream,
                        sink,
                        crate::stream_kernels::detected_backend(),
                    )
                }
            }
        }
    };
}
public_kernel!(L2U8Distance, run_l2u8distance);
public_kernel!(IpU8Distance, run_ipu8distance);
public_kernel!(L2U16Distance, run_l2u16distance);
public_kernel!(
    JLHammingDistance,
    run_jlhammingdistance,
    true,
    cfg!(target_feature = "avx512vpopcntdq")
);
public_kernel!(
    IpI8Distance,
    run_ipi8distance,
    cfg!(target_feature = "dotprod"),
    true
);
public_kernel!(IpI16Distance, run_ipi16distance);
public_kernel!(IpF32Distance, run_ipf32distance);
public_kernel!(L2F32Distance, run_l2f32distance);

/// Asymmetric product-quantization distance: one u8 code selects one f32
/// lookup-table entry. The table already encodes either squared L2 or negative
/// inner product. Uses the ordinary stream's unroll, near prefetch and burst.
#[derive(Debug)]
pub struct PqAdcDistance;
impl DistanceKernelBackend for PqAdcDistance {
    type Storage = u8;
    type Acc = f32;
    const CHUNK_BYTES: usize = 1;
    const QUERY_CHUNK_BYTES: usize = 256 * size_of::<f32>();
    #[inline(always)]
    fn init() -> f32 {
        0.0
    }
    #[inline(always)]
    unsafe fn step(acc: &mut f32, base: *const u8, query: *const u8) {
        // SAFETY: caller supplies one code and its complete f32-aligned table.
        unsafe {
            *acc += query.cast::<f32>().add(base.read() as usize).read();
        }
    }
    #[inline(always)]
    fn reduce(acc: f32) -> f32 {
        acc
    }
    #[inline(always)]
    fn merge(into: &mut f32, src: f32) {
        *into += src;
    }
}
