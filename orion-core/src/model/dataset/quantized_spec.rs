/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Sealed scalar quantization specifications, independent of dataset allocation.
use super::validation;
use crate::Result;
use orion_ann_vector::{DistanceKernelBackend, IpI8Distance, IpI16Distance, L2U8Distance, Metric};

// Storage, kernel and on-disk parameter layout must agree. Keep the set of
// codecs closed; a downstream safe implementation must not weaken this contract.
mod sealed {
    pub trait Sealed {}
    impl Sealed for super::L2U8 {}
    impl Sealed for super::L2U16 {}
    impl Sealed for super::MipsI8 {}
    impl Sealed for super::MipsI16 {}
}

// ─── Trait: per-storage / per-metric specification ────────────────────────

/// Bundle of per-storage + per-metric constants and operations that
/// drive the generic [`super::scalar_dataset::ScalarQuantizedDataset`]. See module docs.
pub trait QuantSpec: sealed::Sealed + 'static {
    /// Underlying storage element (`u8` / `u16` / `i8` / `i16`).
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

    /// Reject parameters that cannot represent a finite positive distance scale.
    fn validate_params(params: &Self::Params) -> Result<()>;

    /// Quantize a single f32 value using the provided params.
    fn quantize_scalar(p: &Self::Params, v: f32) -> Self::Storage;

    /// Quantized-domain distance (already cast to f32). For L2 the
    /// kernel returns sum-of-squares × `1/slope²`-scaled f32; for
    /// MIPS it returns `−Σ a · b × 1/scale²`-scaled f32. Both obey
    /// the "smaller == closer" contract and are directly comparable
    /// against an f32 truth distance scaled via
    /// [`Self::distance_scale_sq`].
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
    /// prefilter loop in `search<Q>` via [`orion_ann_vector::DistanceStream`].
    /// Pairs with the storage / metric: `L2U8Distance` for [`L2U8`],
    /// `IpI8Distance` for [`MipsI8`], `IpI16Distance` for [`MipsI16`].
    type QuantKernel: DistanceKernelBackend<Storage = Self::Storage>;

    /// 32-byte-chunk distance kernel for the **f32 truth** Stage-2
    /// loop. Drives [`orion_ann_vector::DistanceStream`] over the in-memory
    /// f32 base. Same metric as `truth_distance` (sum-of-squares for
    /// L2, negated inner product for MIPS), but in chunked-SIMD
    /// streaming form so Stage-2 enjoys the same pipelined-prefetch
    /// benefits as Stage-1.
    type TruthKernel: DistanceKernelBackend<Storage = f32>;
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

    /// Twin of [`Self::from_range`] for u16 storage. Maps the input range
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
        let x = ((v * self.slope).round() as i64).saturating_sub(self.offset as i64);
        x.clamp(0, 255) as u8
    }

    /// Quantize a single f32 into the u16 range. Mirrors
    /// [`Self::quantize_scalar`] but with the u16 clamp.
    #[inline]
    pub fn quantize_scalar_u16(&self, v: f32) -> u16 {
        let x = ((v * self.slope).round() as i64).saturating_sub(self.offset as i64);
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
/// `orion_ann_vector::distance_l2_vector_u8` (`vabd_u8 + vmlal_u8`).
#[derive(Debug, Clone, Copy)]
pub struct L2U8;

impl QuantSpec for L2U8 {
    fn validate_params(params: &Self::Params) -> Result<()> {
        validation::positive_scale(params.slope)
    }
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
        orion_ann_vector::distance_l2_vector_u8::<N>(aa, bb)
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
        orion_ann_vector::distance_l2_vector_u8_batch4::<N>(v0, v1, v2, v3, qq)
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.slope * p.slope
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        orion_ann_vector::distance_l2_vector_f32::<N>(query, vertex)
    }

    type QuantKernel = L2U8Distance;
    type TruthKernel = orion_ann_vector::L2F32Distance;
}

/// L2 squared-Euclidean on **u16** quantized base. Precise
/// PQ-admission tier of the L2 cascade — 256× finer per-dim
/// resolution than [`L2U8`] at the cost of 2× storage. Distance
/// kernel goes through [`orion_ann_vector::L2U16Distance`] which has the
/// same `DistanceStream` lookahead-prefetch pipeline as the u8 path.
#[derive(Debug, Clone, Copy)]
pub struct L2U16;

impl QuantSpec for L2U16 {
    fn validate_params(params: &Self::Params) -> Result<()> {
        validation::positive_scale(params.slope)
    }
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
            use orion_ann_vector::DistanceKernelBackend;
            let mut acc = orion_ann_vector::L2U16Distance::init();
            let chunks = N / 16; // 16 u16 per 32-byte chunk
            for i in 0..chunks {
                let off = i * 16;
                orion_ann_vector::L2U16Distance::step(&mut acc, a.add(off), b.add(off));
            }
            orion_ann_vector::L2U16Distance::reduce(acc)
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
        orion_ann_vector::distance_l2_vector_f32::<N>(query, vertex)
    }

    type QuantKernel = orion_ann_vector::L2U16Distance;
    type TruthKernel = orion_ann_vector::L2F32Distance;
}

/// MIPS-i8: 8-bit symmetric inner product. Kernel
/// `orion_ann_vector::distance_ip_vector_i8` (`vmull_s8 + vpadalq_s16`).
#[derive(Debug, Clone, Copy)]
pub struct MipsI8;

impl QuantSpec for MipsI8 {
    fn validate_params(params: &Self::Params) -> Result<()> {
        validation::positive_scale(params.scale)
    }
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
        orion_ann_vector::distance_ip_vector_i8::<N>(aa, bb) as f32
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
        let r = orion_ann_vector::distance_ip_vector_i8_batch4::<N>(v0, v1, v2, v3, qq);
        [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.scale * p.scale
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        orion_ann_vector::distance_ip_vector_f32::<N>(query, vertex)
    }

    type QuantKernel = IpI8Distance;
    type TruthKernel = orion_ann_vector::IpF32Distance;
}

/// MIPS-i16: 16-bit symmetric inner product (PA's
/// `-quantize_bits 16 -quantize_mode 1` recipe). Kernel
/// `orion_ann_vector::distance_ip_vector_i16` (`vmull_s16 + vpadalq_s32`).
#[derive(Debug, Clone, Copy)]
pub struct MipsI16;

impl QuantSpec for MipsI16 {
    fn validate_params(params: &Self::Params) -> Result<()> {
        validation::positive_scale(params.scale)
    }
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
        orion_ann_vector::distance_ip_vector_i16::<N>(aa, bb) as f32
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
        let r = orion_ann_vector::distance_ip_vector_i16_batch4::<N>(v0, v1, v2, v3, qq);
        [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]
    }

    #[inline]
    fn distance_scale_sq(p: &Self::Params) -> f32 {
        p.scale * p.scale
    }

    #[inline]
    fn truth_distance<const N: usize>(query: &[f32; N], vertex: &[f32; N]) -> f32 {
        orion_ann_vector::distance_ip_vector_f32::<N>(query, vertex)
    }

    type QuantKernel = IpI16Distance;
    type TruthKernel = orion_ann_vector::IpF32Distance;
}
