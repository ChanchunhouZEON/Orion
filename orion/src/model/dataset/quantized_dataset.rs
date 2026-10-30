/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Unified quantized base store for both **L2 prefilter** and **MIPS
//! beam** search paths. Replaces the earlier triple of standalone
//! types — `QuantizedDataset` (u8 / L2), `QuantizedDatasetMips<i8>`,
//! and `QuantizedDatasetMips<i16>` — with a single generic
//! [`QuantizedDataset<Q>`] parameterised by a [`QuantSpec`]
//! marker that bundles all the per-storage / per-metric differences:
//!
//! - Storage type (`u8` / `i8` / `i16`).
//! - Quantization scheme (affine `(slope, offset)` for L2-u8;
//!   symmetric `scale` for MIPS-i8 / MIPS-i16).
//! - NEON kernel (`distance_l2_vector_u8` / `distance_ip_vector_i8` /
//!   `distance_ip_vector_i16`).
//! - Disk format magic (`QDS` / `QDM8` / `QDM6`).
//! - Per-vertex stride alignment (32 elements for u8/i8 / 16 for i16
//!   — both yield a 32-byte stride, the SIMD / cache-line boundary).
//! - Default prefetch lookahead per the MSHR-depth-vs-vertex-lines
//!   sweet spot.
//! - The truth metric (`Metric::L2` / `Metric::Mips`) for the f32
//!   second-stage compare in the unified `search` path.
//!
//! The three `QuantSpec` impls are zero-sized marker types
//! [`L2U8`], [`MipsI8`], [`MipsI16`]. They never get instantiated as
//! values; everything goes through associated items / functions, so
//! the compiler monomorphises a clean specialization for each.
//!
//! ## Alignment
//!
//! The base buffer is allocated through `AlignedBoxWithSlice` at
//! 32-byte alignment, with each vertex's slot rounded up to a
//! storage-element-count multiple of `Q::ALIGN_ELEMS`. The padding
//! bytes are zero (`alloc_zeroed` inside `AlignedBoxWithSlice::new`)
//! and never read by the kernels — the NEON main loop processes
//! whole vector lanes, scalar tails handle the residual within `N`.
//! On-disk format is **packed** (no padding) for portability;
//! padding is reintroduced at load time.

use super::rows::F32Rows;
use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use vector::{DistanceFn, IpI8Distance, IpI16Distance, L2U8Distance, Metric};

// ─── Trait: per-storage / per-metric specification ────────────────────────

/// Bundle of per-storage + per-metric constants and operations that
/// drive the generic [`QuantizedDataset`]. See module docs.
pub trait QuantSpec: 'static {
    /// Underlying storage element (`u8` / `i8` / `i16`).
    type Storage: bytemuck::Pod + Copy + Default + Send + Sync + 'static;
    /// Quantization parameter struct. L2-u8 uses
    /// [`QuantParamsL2`] (slope + offset, two-tuple affine);
    /// MIPS-i8 / MIPS-i16 use [`QuantParamsMips`] (single scale,
    /// symmetric).
    type Params: bytemuck::Pod + Send + Sync + std::fmt::Debug;

    /// Disk format magic. Distinct per spec so loading the wrong
    /// sidecar fails fast.
    const MAGIC: u32;
    /// File extension hint for the sidecar (no leading dot).
    const FILE_EXT: &'static str;
    /// Short label for log lines.
    const LABEL: &'static str;
    /// Per-vertex stride alignment in *elements*. Picked so that
    /// `STRIDE * sizeof(Self::Storage) % 32 == 0`. 32 for u8/i8; 16
    /// for i16 — both yield 32-byte stride.
    const ALIGN_ELEMS: usize;
    /// Default prefetch lookahead (in elements) for the batch loop.
    /// Tuned per the MSHR-depth-vs-vertex-lines sweet spot.
    const PF_BATCH_DEFAULT: usize;
    /// Truth-distance metric used by the f32 second stage in the
    /// unified search path.
    const METRIC: Metric;

    /// Build params from a flat `[f32]` slice (the f32 base
    /// dataset's value range / abs-max).
    fn build_params(slice: &[f32]) -> Self::Params;

    /// Quantize a single f32 value using the provided params.
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage;

    /// Quantized-domain distance (already cast to f32). For L2 the
    /// kernel returns sum-of-squares × `1/slope²`-scaled f32; for
    /// MIPS it returns `−Σ a · b × 1/scale²`-scaled f32. Both obey
    /// the "smaller == closer" contract and are directly comparable
    /// against an f32 truth distance scaled via
    /// [`distance_scale_sq`].
    ///
    /// # Safety
    /// Both pointers must be valid `[Self::Storage; N]` regions.
    unsafe fn distance<const N: usize>(a: *const Self::Storage, b: *const Self::Storage) -> f32;

    /// 4-way batched quantized distance.
    ///
    /// # Safety
    /// All five pointers must be valid `[Self::Storage; N]` regions.
    unsafe fn distance_batch4<const N: usize>(
        a0: *const Self::Storage,
        a1: *const Self::Storage,
        a2: *const Self::Storage,
        a3: *const Self::Storage,
        q: *const Self::Storage,
    ) -> [f32; 4];

    /// `distance_scale_sq(p)` is the factor relating quantized
    /// distance to the f32 truth distance:
    ///   `quantized_dist ≈ f32_truth_dist * distance_scale_sq(p)`.
    /// L2: `slope²`. MIPS: `scale²`. Search code uses this to scale
    /// `pq_worst` (an f32 truth distance) into the quantized
    /// comparison space for the prefilter threshold.
    fn distance_scale_sq(p: &Self::Params) -> f32;

    /// f32 truth distance between a query and a base vertex. Used by
    /// the second stage in `search<Q>` (after the quantized prefilter).
    /// `smaller == closer`. L2: `Σ (q-v)²`. MIPS: `−Σ q·v`.
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32;

    /// 32-byte-chunk distance kernel that drives the Stage-1
    /// prefilter loop in `search<Q>` via [`vector::DistanceStream`].
    /// Pairs with the storage / metric: `L2U8Distance` for [`L2U8`],
    /// `IpI8Distance` for [`MipsI8`], `IpI16Distance` for [`MipsI16`].
    type QuantDistanceFn: DistanceFn<Storage = Self::Storage>;

    /// 32-byte-chunk distance kernel for the **f32 truth** Stage-2
    /// loop. Drives [`vector::DistanceStream`] over the in-memory
    /// f32 base. Same metric as `truth_distance` (sum-of-squares for
    /// L2, negated inner product for MIPS), but in chunked-SIMD
    /// streaming form so Stage-2 enjoys the same pipelined-prefetch
    /// benefits as Stage-1.
    type TruthDistanceFn: DistanceFn<Storage = f32>;
}

// ─── Param structs ────────────────────────────────────────────────────────

/// L2-u8 affine quantization: `u8 = clamp(round(v · slope) − offset, 0, 255)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantParamsL2 {
    pub slope: f32,
    pub offset: i32,
}

impl QuantParamsL2 {
    pub fn from_range(min_val: f32, max_val: f32) -> Self {
        let range = (max_val - min_val).max(1e-9);
        let slope = 255.0 / range;
        let offset = (min_val * slope).round() as i32;
        Self { slope, offset }
    }

    /// Twin of [`from_range`] for u16 storage. Maps the input range
    /// to `[0, 65535]` (256× finer per-dim resolution than u8). Used
    /// by the [`L2U16`] codec as the precise PQ-admission tier of
    /// the L2 cascade.
    pub fn from_range_u16(min_val: f32, max_val: f32) -> Self {
        let range = (max_val - min_val).max(1e-9);
        let slope = 65535.0 / range;
        let offset = (min_val * slope).round() as i32;
        Self { slope, offset }
    }

    #[inline]
    pub fn quantize_scalar(&self, v: f32) -> u8 {
        let x = (v * self.slope).round() as i32 - self.offset;
        x.clamp(0, 255) as u8
    }

    /// Quantize a single f32 into the u16 range. Mirrors
    /// [`quantize_scalar`] but with the u16 clamp.
    #[inline]
    pub fn quantize_scalar_u16(&self, v: f32) -> u16 {
        let x = (v * self.slope).round() as i32 - self.offset;
        x.clamp(0, 65535) as u16
    }
}

/// MIPS symmetric quantization: `q = round(v · scale)` with `|q| ≤ MAX_Q`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantParamsMips {
    pub scale: f32,
}

impl QuantParamsMips {
    pub fn from_abs_max(abs_max: f32, max_q: f32) -> Self {
        let scale = if abs_max > 0.0 { max_q / abs_max } else { 1.0 };
        Self { scale }
    }
}

// ─── Marker types + trait impls ───────────────────────────────────────────

/// L2 squared-Euclidean prefilter on u8 quantized base. Two-parameter
/// affine quantization (`slope`, `offset`); kernel
/// `vector::distance_l2_vector_u8` (`vabd_u8 + vmlal_u8`).
#[derive(Debug, Clone, Copy)]
pub struct L2U8;

impl QuantSpec for L2U8 {
    type Storage = u8;
    type Params = QuantParamsL2;
    const MAGIC: u32 = 0x5144_5300; // "QDS\0"
    const FILE_EXT: &'static str = "qds";
    const LABEL: &'static str = "u8/L2";
    const ALIGN_ELEMS: usize = 32; // 32 × 1 B = 32 B stride
    const PF_BATCH_DEFAULT: usize = 8;
    const METRIC: Metric = Metric::L2;

    fn build_params(slice: &[f32]) -> Self::Params {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        QuantParamsL2::from_range(min_val, max_val)
    }

    #[inline]
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage {
        p.quantize_scalar(v)
    }

    #[inline]
    unsafe fn distance<const N: usize>(a: *const u8, b: *const u8) -> f32 {
        let aa = unsafe { &*(a as *const [u8; N]) };
        let bb = unsafe { &*(b as *const [u8; N]) };
        vector::distance_l2_vector_u8::<N>(aa, bb)
    }

    #[inline]
    unsafe fn distance_batch4<const N: usize>(
        a0: *const u8,
        a1: *const u8,
        a2: *const u8,
        a3: *const u8,
        q: *const u8,
    ) -> [f32; 4] {
        let v0 = unsafe { &*(a0 as *const [u8; N]) };
        let v1 = unsafe { &*(a1 as *const [u8; N]) };
        let v2 = unsafe { &*(a2 as *const [u8; N]) };
        let v3 = unsafe { &*(a3 as *const [u8; N]) };
        let qq = unsafe { &*(q as *const [u8; N]) };
        vector::distance_l2_vector_u8_batch4::<N>(v0, v1, v2, v3, qq)
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.slope * p.slope
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        vector::distance_l2_vector_f32::<N>(query, vertex)
    }

    type QuantDistanceFn = L2U8Distance;
    type TruthDistanceFn = vector::L2F32Distance;
}

/// L2 squared-Euclidean on **u16** quantized base. Precise
/// PQ-admission tier of the L2 cascade — 256× finer per-dim
/// resolution than [`L2U8`] at the cost of 2× storage. Distance
/// kernel goes through [`vector::L2U16Distance`] which has the
/// same `DistanceStream` lookahead-prefetch pipeline as the u8 path.
#[derive(Debug, Clone, Copy)]
pub struct L2U16;

impl QuantSpec for L2U16 {
    type Storage = u16;
    type Params = QuantParamsL2;
    const MAGIC: u32 = 0x5144_5336; // "QDS6"
    const FILE_EXT: &'static str = "qds6";
    const LABEL: &'static str = "u16/L2";
    const ALIGN_ELEMS: usize = 16; // 16 × 2 B = 32 B stride
    const PF_BATCH_DEFAULT: usize = 8;
    const METRIC: Metric = Metric::L2;

    fn build_params(slice: &[f32]) -> Self::Params {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        QuantParamsL2::from_range_u16(min_val, max_val)
    }

    #[inline]
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage {
        p.quantize_scalar_u16(v)
    }

    /// Single-shot distance via the streaming kernel: iterate
    /// `L2U16Distance::step` over 32-byte (16 u16-lane) chunks of the
    /// full N-element vector, reduce once at the end. The streaming
    /// kernel already inherits the lookahead-prefetch pipeline so
    /// hot-path callers (entry distance, calibration) get the same
    /// memory-pipelining as the batched DistanceStream loop.
    #[inline]
    unsafe fn distance<const N: usize>(a: *const u16, b: *const u16) -> f32 {
        unsafe {
            use vector::DistanceFn;
            let mut acc = vector::L2U16Distance::init();
            let chunks = N / 16; // 16 u16 per 32-byte chunk
            for i in 0..chunks {
                let off = i * 16;
                vector::L2U16Distance::step(&mut acc, a.add(off), b.add(off));
            }
            vector::L2U16Distance::reduce(acc)
        }
    }

    /// 4-way batched single-shot: four serial calls to
    /// [`distance`](Self::distance). The hot path uses
    /// `DistanceStream<L2U16Distance>` for true 4-way ILP — this
    /// helper is only invoked from the cold paths (entry distance
    /// batch, calibration warmup) so an unrolled 4-way kernel isn't
    /// worth the duplication.
    #[inline]
    unsafe fn distance_batch4<const N: usize>(
        a0: *const u16,
        a1: *const u16,
        a2: *const u16,
        a3: *const u16,
        q: *const u16,
    ) -> [f32; 4] {
        unsafe {
            [
                Self::distance::<N>(a0, q),
                Self::distance::<N>(a1, q),
                Self::distance::<N>(a2, q),
                Self::distance::<N>(a3, q),
            ]
        }
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.slope * p.slope
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        vector::distance_l2_vector_f32::<N>(query, vertex)
    }

    type QuantDistanceFn = vector::L2U16Distance;
    type TruthDistanceFn = vector::L2F32Distance;
}

/// MIPS-i8: 8-bit symmetric inner product. Kernel
/// `vector::distance_ip_vector_i8` (`vmull_s8 + vpadalq_s16`).
#[derive(Debug, Clone, Copy)]
pub struct MipsI8;

impl QuantSpec for MipsI8 {
    type Storage = i8;
    type Params = QuantParamsMips;
    const MAGIC: u32 = 0x5144_4D38; // "QDM8"
    const FILE_EXT: &'static str = "qdm8";
    const LABEL: &'static str = "i8/MIPS";
    const ALIGN_ELEMS: usize = 32; // 32 × 1 B = 32 B stride
    const PF_BATCH_DEFAULT: usize = 4;
    const METRIC: Metric = Metric::Cosine;

    fn build_params(slice: &[f32]) -> Self::Params {
        let mut abs_max = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > abs_max {
                abs_max = av;
            }
        }
        QuantParamsMips::from_abs_max(abs_max, 127.0)
    }

    #[inline]
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage {
        (v * p.scale).round().clamp(-127.0, 127.0) as i8
    }

    #[inline]
    unsafe fn distance<const N: usize>(a: *const i8, b: *const i8) -> f32 {
        let aa = unsafe { &*(a as *const [i8; N]) };
        let bb = unsafe { &*(b as *const [i8; N]) };
        vector::distance_ip_vector_i8::<N>(aa, bb) as f32
    }

    #[inline]
    unsafe fn distance_batch4<const N: usize>(
        a0: *const i8,
        a1: *const i8,
        a2: *const i8,
        a3: *const i8,
        q: *const i8,
    ) -> [f32; 4] {
        let v0 = unsafe { &*(a0 as *const [i8; N]) };
        let v1 = unsafe { &*(a1 as *const [i8; N]) };
        let v2 = unsafe { &*(a2 as *const [i8; N]) };
        let v3 = unsafe { &*(a3 as *const [i8; N]) };
        let qq = unsafe { &*(q as *const [i8; N]) };
        let r = vector::distance_ip_vector_i8_batch4::<N>(v0, v1, v2, v3, qq);
        [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.scale * p.scale
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        vector::distance_ip_vector_f32::<N>(query, vertex)
    }

    type QuantDistanceFn = IpI8Distance;
    type TruthDistanceFn = vector::IpF32Distance;
}

/// MIPS-i16: 16-bit symmetric inner product (PA's
/// `-quantize_bits 16 -quantize_mode 1` recipe). Kernel
/// `vector::distance_ip_vector_i16` (`vmull_s16 + vpadalq_s32`).
#[derive(Debug, Clone, Copy)]
pub struct MipsI16;

impl QuantSpec for MipsI16 {
    type Storage = i16;
    type Params = QuantParamsMips;
    const MAGIC: u32 = 0x5144_4D36; // "QDM6"
    const FILE_EXT: &'static str = "qdm16";
    const LABEL: &'static str = "i16/MIPS";
    const ALIGN_ELEMS: usize = 16; // 16 × 2 B = 32 B stride
    const PF_BATCH_DEFAULT: usize = 4;
    const METRIC: Metric = Metric::Cosine;

    fn build_params(slice: &[f32]) -> Self::Params {
        let mut abs_max = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > abs_max {
                abs_max = av;
            }
        }
        QuantParamsMips::from_abs_max(abs_max, 32767.0)
    }

    #[inline]
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage {
        (v * p.scale).round().clamp(-32767.0, 32767.0) as i16
    }

    #[inline]
    unsafe fn distance<const N: usize>(a: *const i16, b: *const i16) -> f32 {
        let aa = unsafe { &*(a as *const [i16; N]) };
        let bb = unsafe { &*(b as *const [i16; N]) };
        vector::distance_ip_vector_i16::<N>(aa, bb) as f32
    }

    #[inline]
    unsafe fn distance_batch4<const N: usize>(
        a0: *const i16,
        a1: *const i16,
        a2: *const i16,
        a3: *const i16,
        q: *const i16,
    ) -> [f32; 4] {
        let v0 = unsafe { &*(a0 as *const [i16; N]) };
        let v1 = unsafe { &*(a1 as *const [i16; N]) };
        let v2 = unsafe { &*(a2 as *const [i16; N]) };
        let v3 = unsafe { &*(a3 as *const [i16; N]) };
        let qq = unsafe { &*(q as *const [i16; N]) };
        let r = vector::distance_ip_vector_i16_batch4::<N>(v0, v1, v2, v3, qq);
        [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.scale * p.scale
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        vector::distance_ip_vector_f32::<N>(query, vertex)
    }

    type QuantDistanceFn = IpI16Distance;
    type TruthDistanceFn = vector::IpF32Distance;
}

// ─── Generic dataset ──────────────────────────────────────────────────────

/// Quantized base store. `Q` selects the storage + metric (one of
/// [`L2U8`], [`MipsI8`], [`MipsI16`]); `N` is the vector dimension.
#[derive(Debug)]
pub struct QuantizedDataset<Q: QuantSpec> {
    pub data: AlignedBoxWithSlice<Q::Storage>,
    pub params: Q::Params,
    pub n: usize,
    pub dimension: usize,
    pub stride: usize,
}

impl<Q: QuantSpec> QuantizedDataset<Q> {
    /// Per-vertex element span, rounded up so each vertex's byte
    /// footprint ends on the selected distance kernel’s block boundary.
    pub fn from_f32_dataset(ds: &impl F32Rows) -> Self {
        let dimension = ds.dimension();
        let n = ds.len();
        assert!(dimension > 0 && n > 0);
        assert!(dimension <= Q::QuantDistanceFn::MAX_COORDINATES);
        let layout = vector::PaddedLayout::for_kernel::<Q::QuantDistanceFn>(dimension)
            .expect("valid quantized row layout");
        let stride = layout.stride_elements();
        // Padding is not a sample: only logical values may fit the quantizer.
        let (min, max) = ds.range();
        let params = Q::build_params(&[min, max]);
        let mut data = AlignedBoxWithSlice::<Q::Storage>::new(n * stride, layout.alignment())
            .expect("allocate aligned quantized buffer");
        use rayon::prelude::*;
        data.as_mut_slice()
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(id, out)| {
                for (dst, &value) in out.iter_mut().zip(ds.row(id)) {
                    *dst = Q::quantize_scalar(&params, value);
                }
            });
        Self {
            data,
            params,
            n,
            dimension,
            stride,
        }
    }

    /// # Safety
    /// UB if `id >= n`. Returned pointer is 32-byte aligned.
    #[inline]
    pub unsafe fn get_vertex_unchecked(&self, id: u32) -> &[Q::Storage] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(id as usize * self.stride),
                self.dimension,
            )
        }
    }

    /// Quantize only the logical coordinates. `qdist` accepts this packed
    /// form, while search sessions use the padded form to avoid per-hop copies.
    pub fn quantize_query(&self, query: &[f32]) -> Vec<Q::Storage> {
        assert_eq!(query.len(), self.dimension);
        query
            .iter()
            .map(|&v| Q::quantize_scalar(&self.params, v))
            .collect()
    }

    /// Quantize a query into a heap-allocated `Vec<Storage>` of size
    /// [`self.stride`] — the same per-vertex element span as the base
    /// buffer, with the trailing `STRIDE - N` slots zero-padded so
    /// the kernel sees the same byte footprint per vertex on both
    /// sides. This is the form [`vector::DistanceStream`] expects:
    /// it reads complete kernel blocks from
    /// both the base and the query, so any byte the base touches must
    /// also be present (and zero) in the query. Allocated once per
    /// `search<Q>` call, reused across every hop.
    pub fn quantize_query_padded(&self, query: &[f32]) -> Vec<Q::Storage> {
        assert_eq!(query.len(), self.dimension);
        let mut out = vec![Q::Storage::default(); self.stride];
        for i in 0..self.dimension {
            out[i] = Q::quantize_scalar(&self.params, query[i]);
        }
        // out[N..STRIDE] retains its `Default` value (0 for i8/i16/u8).
        out
    }

    /// Quantized distance between vertex `id` and the query (already
    /// cast to f32, "smaller == closer"). Caller should compare against
    /// `f32_threshold * Q::distance_scale_sq(self.params) * Q_SLACK` to
    /// stay in the same comparison space.
    #[inline]
    pub unsafe fn qdist(&self, id: u32, q: &[Q::Storage]) -> f32 {
        assert!(q.len() == self.dimension || q.len() == self.stride);
        let padded;
        let query = if q.len() == self.stride {
            q
        } else {
            padded = {
                let mut v = vec![Q::Storage::default(); self.stride];
                v[..self.dimension].copy_from_slice(q);
                v
            };
            &padded
        };
        let ids = [id];
        let mut distance = 0.;
        unsafe {
            vector::DistanceStream::<Q::QuantDistanceFn>::new(
                self.data.as_ptr().cast(),
                self.stride * std::mem::size_of::<Q::Storage>(),
                self.stride * std::mem::size_of::<Q::Storage>(),
                query.as_ptr().cast(),
                &ids,
                1,
            )
            .run(|_, d| distance = d);
        }
        distance
    }

    /// 4-way batched quantized distance.
    #[inline]
    pub unsafe fn qdist4(&self, ids: [u32; 4], q: &[Q::Storage]) -> [f32; 4] {
        ids.map(|id| unsafe { self.qdist(id, q) })
    }

    /// Convert `f32_pq_worst` into the quantized distance scale, ready
    /// for direct comparison with `qdist` / `qdist4` outputs.
    #[inline]
    pub fn pq_worst_to_quantized(&self, f32_pq_worst: f32) -> f32 {
        f32_pq_worst * Q::distance_scale_sq(&self.params)
    }

    #[inline]
    pub fn prefetch_vector(&self, id: u32) {
        unsafe {
            let v = self.get_vertex_unchecked(id);
            vector::prefetch_vector(v);
        }
    }

    /// Header layout (24 B):
    ///   [u32 magic = Q::MAGIC]
    ///   [u32 version = 1]
    ///   [u32 n][u32 dim = N]
    ///   [Q::Params packed into 8 B]   ← spec-specific
    /// Body: packed `n × N` storage elements (no STRIDE padding).
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> ANNResult<()> {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&Q::MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.n as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(self.dimension as u32).to_le_bytes());
        // Fixed POD layout preserves the legacy header without unaligned reads.
        let params = bytemuck::bytes_of(&self.params);
        assert!(
            params.len() <= 8,
            "quantization parameters exceed the header"
        );
        hdr[16..16 + params.len()].copy_from_slice(params);
        w.write_all(&hdr)?;
        let elem_bytes = std::mem::size_of::<Q::Storage>();
        let base = self.data.as_slice();
        for id in 0..self.n {
            let off = id * self.stride;
            let vertex_bytes = unsafe {
                std::slice::from_raw_parts(
                    base.as_ptr().add(off) as *const u8,
                    self.dimension * elem_bytes,
                )
            };
            w.write_all(vertex_bytes)?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> ANNResult<Self> {
        use std::io::Read;
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr)?;
        if u32::from_le_bytes(hdr[4..8].try_into().unwrap()) != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported sidecar version",
            )
            .into());
        }
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != Q::MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = {})",
                    Q::MAGIC,
                    Q::LABEL
                ),
            )
            .into());
        }
        let n = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if dim == 0 || n == 0 || dim > Q::QuantDistanceFn::MAX_COORDINATES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid quantized shape",
            )
            .into());
        }
        let expected = (n as u64)
            .checked_mul(dim as u64)
            .and_then(|x| x.checked_mul(std::mem::size_of::<Q::Storage>() as u64))
            .and_then(|x| x.checked_add(24));
        if expected != Some(r.get_ref().metadata()?.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "quantized payload length does not match its shape",
            )
            .into());
        }
        let layout = vector::PaddedLayout::for_kernel::<Q::QuantDistanceFn>(dim)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let stride = layout.stride_elements();
        let param_bytes = std::mem::size_of::<Q::Params>();
        if param_bytes > 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "quantization parameters exceed the header",
            )
            .into());
        }
        let params: Q::Params = bytemuck::pod_read_unaligned(&hdr[16..16 + param_bytes]);
        let data = super::rows::read_rows::<Q::Storage>(&mut r, n, stride, dim)?;
        Ok(Self {
            data,
            params,
            n,
            dimension: dim,
            stride,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann::model::InmemDataset;

    fn make_ds<const N: usize>(num_points: usize) -> InmemDataset<f32, N> {
        let mut ds = InmemDataset::<f32, N>::new(num_points, 1.0).unwrap();
        let s = ds.data.as_mut_slice();
        for (i, v) in s.iter_mut().enumerate() {
            *v = ((i % 23) as f32) * 0.1 - 0.5;
        }
        ds
    }

    // ─── QuantParamsL2 ────────────────────────────────────────────────

    #[test]
    fn l2_params_from_range_round_trip_u8() {
        let p = QuantParamsL2::from_range(-1.0, 1.0);
        assert_eq!(p.quantize_scalar(-1.0), 0);
        assert_eq!(p.quantize_scalar(1.0), 255);
        let mid = p.quantize_scalar(0.0);
        assert!(mid > 120 && mid < 136);
    }

    #[test]
    fn l2_params_from_range_zero_range_safe() {
        // Degenerate input: min == max. `from_range` is expected to
        // clamp the divisor to ε so neither construction nor quantize
        // panics — the returned u8 value itself is implementation-
        // defined under range collapse, so the test only asserts that
        // execution reaches the end of the function.
        let p = QuantParamsL2::from_range(0.5, 0.5);
        let _ = p.quantize_scalar(0.5);
    }

    #[test]
    fn l2_params_from_range_u16_endpoints() {
        let p = QuantParamsL2::from_range_u16(-1.0, 1.0);
        assert_eq!(p.quantize_scalar_u16(-1.0), 0);
        assert_eq!(p.quantize_scalar_u16(1.0), 65535);
    }

    #[test]
    fn l2_params_quantize_clamps() {
        let p_u8 = QuantParamsL2::from_range(0.0, 1.0);
        assert_eq!(p_u8.quantize_scalar(-10.0), 0);
        assert_eq!(p_u8.quantize_scalar(100.0), 255);
        let p_u16 = QuantParamsL2::from_range_u16(0.0, 1.0);
        assert_eq!(p_u16.quantize_scalar_u16(-10.0), 0);
        assert_eq!(p_u16.quantize_scalar_u16(100.0), 65535);
    }

    // ─── QuantParamsMips ──────────────────────────────────────────────

    #[test]
    fn mips_params_from_abs_max_basic() {
        let p = QuantParamsMips::from_abs_max(1.0, 127.0);
        assert!((p.scale - 127.0).abs() < 1e-6);
    }

    #[test]
    fn mips_params_from_abs_max_zero_safe() {
        let p = QuantParamsMips::from_abs_max(0.0, 127.0);
        assert!(p.scale.is_finite());
    }

    // ─── QuantizedDataset<L2U8> ──────────────────────────────────────

    #[test]
    fn build_l2u8_dataset() {
        let ds = make_ds::<8>(16);
        let qds = QuantizedDataset::<L2U8>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 16);
        let q = [0.0f32; 8];
        let qq = qds.quantize_query(&q);
        assert_eq!(qq.len(), 8);
        let pad = qds.quantize_query_padded(&q);
        assert_eq!(pad.len(), qds.stride);
    }

    #[test]
    fn pq_worst_to_quantized_scales_with_slope_sq() {
        let ds = make_ds::<4>(4);
        let qds = QuantizedDataset::<L2U8>::from_f32_dataset(&ds);
        let f32_w = 1.0f32;
        let q_w = qds.pq_worst_to_quantized(f32_w);
        let expected = qds.params.slope * qds.params.slope * f32_w;
        assert!((q_w - expected).abs() < 1e-3);
    }

    // ─── Save / load round trips ─────────────────────────────────────

    #[test]
    fn l2u8_save_load_round_trip() {
        let tmp = std::env::temp_dir().join("orion_qds_l2u8_test.bin");
        let _ = std::fs::remove_file(&tmp);
        let ds = make_ds::<4>(8);
        let qds = QuantizedDataset::<L2U8>::from_f32_dataset(&ds);
        qds.save(&tmp).unwrap();
        let loaded = QuantizedDataset::<L2U8>::load(&tmp).unwrap();
        assert_eq!(loaded.n, qds.n);
        assert!((loaded.params.slope - qds.params.slope).abs() < 1e-6);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn build_mips_i8_dataset() {
        let ds = make_ds::<8>(4);
        let qds = QuantizedDataset::<MipsI8>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 4);
        assert_eq!((qds.stride) % 32, 0);
    }

    #[test]
    fn build_mips_i16_dataset() {
        let ds = make_ds::<8>(4);
        let qds = QuantizedDataset::<MipsI16>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 4);
        assert_eq!((qds.stride * 2) % 32, 0);
    }

    #[test]
    fn build_l2_u16_dataset() {
        let ds = make_ds::<8>(8);
        let qds = QuantizedDataset::<L2U16>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 8);
    }
}
