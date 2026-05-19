/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Unified quantized base store for both **L2 prefilter** and **MIPS
//! beam** search paths. Replaces the earlier triple of standalone
//! types — `QuantizedDataset` (u8 / L2), `QuantizedDatasetMips<i8>`,
//! and `QuantizedDatasetMips<i16>` — with a single generic
//! [`QuantizedDataset<Q, N>`] parameterised by a [`QuantSpec`]
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

use diskann::common::AlignedBoxWithSlice;
use diskann::model::InmemDataset;
use vector::{
    DistanceFn, FullPrecisionDistance, IpI8Distance, IpI16Distance, L2U8Distance, Metric,
};

// ─── Trait: per-storage / per-metric specification ────────────────────────

/// Bundle of per-storage + per-metric constants and operations that
/// drive the generic [`QuantizedDataset`]. See module docs.
pub trait QuantSpec: 'static {
    /// Underlying storage element (`u8` / `i8` / `i16`).
    type Storage: Copy + Default + Send + Sync + 'static;
    /// Quantization parameter struct. L2-u8 uses
    /// [`QuantParamsL2`] (slope + offset, two-tuple affine);
    /// MIPS-i8 / MIPS-i16 use [`QuantParamsMips`] (single scale,
    /// symmetric).
    type Params: Copy + Send + Sync + std::fmt::Debug;

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

    /// Whether the query must be L2-normalized before search. Set on
    /// MIPS specs (cosine search assumes both sides on the unit
    /// sphere); off for L2.
    const NORMALIZE_QUERY: bool;

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
#[derive(Debug, Clone, Copy)]
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

    #[inline]
    pub fn quantize_scalar(&self, v: f32) -> u8 {
        let x = (v * self.slope).round() as i32 - self.offset;
        x.clamp(0, 255) as u8
    }
}

/// MIPS symmetric quantization: `q = round(v · scale)` with `|q| ≤ MAX_Q`.
#[derive(Debug, Clone, Copy)]
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

    const NORMALIZE_QUERY: bool = false;

    type QuantDistanceFn = L2U8Distance;
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

    const NORMALIZE_QUERY: bool = true;

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

    const NORMALIZE_QUERY: bool = true;

    type QuantDistanceFn = IpI16Distance;
    type TruthDistanceFn = vector::IpF32Distance;
}

// ─── Generic dataset ──────────────────────────────────────────────────────

/// Quantized base store. `Q` selects the storage + metric (one of
/// [`L2U8`], [`MipsI8`], [`MipsI16`]); `N` is the vector dimension.
#[derive(Debug)]
pub struct QuantizedDataset<Q: QuantSpec, const N: usize> {
    pub data: AlignedBoxWithSlice<Q::Storage>,
    pub params: Q::Params,
    pub n: usize,
}

impl<Q: QuantSpec, const N: usize> QuantizedDataset<Q, N> {
    /// Per-vertex element span, rounded up so each vertex's byte
    /// footprint ends on a 32-byte boundary.
    pub const STRIDE: usize = (N + Q::ALIGN_ELEMS - 1) & !(Q::ALIGN_ELEMS - 1);

    pub fn from_f32_dataset(ds: &InmemDataset<f32, N>) -> Self
    where
        [f32; N]: FullPrecisionDistance<f32, N>,
    {
        let n = ds.num_active_pts;
        let slice = ds.get_data();
        let len = n * N;
        let params = Q::build_params(&slice[..len]);

        let mut data = AlignedBoxWithSlice::<Q::Storage>::new(n * Self::STRIDE, 32)
            .expect("allocate aligned quantized buffer");

        use rayon::prelude::*;
        data.as_mut_slice()
            .par_chunks_mut(Self::STRIDE)
            .zip(slice[..len].par_chunks(N))
            .for_each(|(out, inp)| {
                for i in 0..N {
                    out[i] = Q::quantize_scalar(&params, inp[i]);
                }
                // Trailing out[N..STRIDE] is already zero (alloc_zeroed).
            });

        Self { data, params, n }
    }

    /// # Safety
    /// UB if `id >= n`. Returned pointer is 32-byte aligned.
    #[inline]
    pub unsafe fn get_vertex_unchecked(&self, id: u32) -> &[Q::Storage; N] {
        let ptr = unsafe { self.data.as_ptr().add(id as usize * Self::STRIDE) as *const [Q::Storage; N] };
        unsafe { &*ptr }
    }

    /// Quantize a query into a stack-allocated `[Storage; N]`. Used
    /// only by legacy callers (`qdist`, `qdist4`) that read exactly
    /// `N` elements and ignore the padding.
    pub fn quantize_query(&self, query: &[f32; N]) -> [Q::Storage; N] {
        let mut out = [Q::Storage::default(); N];
        for i in 0..N {
            out[i] = Q::quantize_scalar(&self.params, query[i]);
        }
        out
    }

    /// Quantize a query into a heap-allocated `Vec<Storage>` of size
    /// [`Self::STRIDE`] — the same per-vertex element span as the base
    /// buffer, with the trailing `STRIDE - N` slots zero-padded so
    /// the kernel sees the same byte footprint per vertex on both
    /// sides. This is the form [`vector::DistanceStream`] expects:
    /// it reads `stride_bytes / 32` chunks of 32-byte windows from
    /// both the base and the query, so any byte the base touches must
    /// also be present (and zero) in the query. Allocated once per
    /// `search<Q>` call, reused across every hop.
    pub fn quantize_query_padded(&self, query: &[f32; N]) -> Vec<Q::Storage> {
        let mut out = vec![Q::Storage::default(); Self::STRIDE];
        for i in 0..N {
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
    pub unsafe fn qdist(&self, id: u32, q: &[Q::Storage; N]) -> f32 {
        let v = unsafe { self.get_vertex_unchecked(id) };
        unsafe { Q::distance::<N>(v.as_ptr(), q.as_ptr()) }
    }

    /// 4-way batched quantized distance.
    #[inline]
    pub unsafe fn qdist4(&self, ids: [u32; 4], q: &[Q::Storage; N]) -> [f32; 4] {
        let v0 = unsafe { self.get_vertex_unchecked(ids[0]) };
        let v1 = unsafe { self.get_vertex_unchecked(ids[1]) };
        let v2 = unsafe { self.get_vertex_unchecked(ids[2]) };
        let v3 = unsafe { self.get_vertex_unchecked(ids[3]) };
        unsafe {
            Q::distance_batch4::<N>(
                v0.as_ptr(),
                v1.as_ptr(),
                v2.as_ptr(),
                v3.as_ptr(),
                q.as_ptr(),
            )
        }
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
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&Q::MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.n as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(N as u32).to_le_bytes());
        // Pack params into hdr[16..24]. Both QuantParamsL2 (slope:f32 +
        // offset:i32) and QuantParamsMips (scale:f32 + 4 B pad) fit in
        // 8 bytes, so this is a thin spec-aware copy via unsafe transmute.
        // SAFETY: both Param structs are #[repr(Rust)] but hold POD
        // members totalling ≤ 8 bytes; we transmute via byte-array copy.
        unsafe {
            let pbytes = std::slice::from_raw_parts(
                &self.params as *const Q::Params as *const u8,
                std::mem::size_of::<Q::Params>().min(8),
            );
            hdr[16..16 + pbytes.len()].copy_from_slice(pbytes);
        }
        w.write_all(&hdr)?;
        let elem_bytes = std::mem::size_of::<Q::Storage>();
        let base = self.data.as_slice();
        for id in 0..self.n {
            let off = id * Self::STRIDE;
            let vertex_bytes = unsafe {
                std::slice::from_raw_parts(base.as_ptr().add(off) as *const u8, N * elem_bytes)
            };
            w.write_all(vertex_bytes)?;
        }
        w.flush()
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        use std::io::Read;
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != Q::MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = {})",
                    Q::MAGIC,
                    Q::LABEL
                ),
            ));
        }
        let n = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if dim != N {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("dim mismatch: file={dim} expected={N}"),
            ));
        }
        // SAFETY: see save() — we re-read the same byte layout the
        // matching impl wrote.
        let params: Q::Params = unsafe {
            let mut buf = [0u8; 8];
            let n_bytes = std::mem::size_of::<Q::Params>().min(8);
            buf[..n_bytes].copy_from_slice(&hdr[16..16 + n_bytes]);
            std::ptr::read(buf.as_ptr() as *const Q::Params)
        };
        let mut data = AlignedBoxWithSlice::<Q::Storage>::new(n * Self::STRIDE, 32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        let elem_bytes = std::mem::size_of::<Q::Storage>();
        if Self::STRIDE == N {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_slice().as_mut_ptr() as *mut u8,
                    n * N * elem_bytes,
                )
            };
            r.read_exact(bytes)?;
        } else {
            let base = data.as_mut_slice();
            let mut buf = vec![0u8; N * elem_bytes];
            for id in 0..n {
                r.read_exact(&mut buf)?;
                let off = id * Self::STRIDE;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.as_ptr(),
                        base.as_mut_ptr().add(off) as *mut u8,
                        N * elem_bytes,
                    );
                }
            }
        }
        Ok(Self { data, params, n })
    }
}
