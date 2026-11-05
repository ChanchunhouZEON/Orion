/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **L2 kernel-trick** quantized base: i8 storage + per-vertex
//! squared L2 norm, designed to let `sdot` carry the per-hop distance
//! compute while staying L2-rank-exact.
//!
//! ## Why a separate sidecar from `L2U8`
//!
//! The default L2-Q kernel ([`orion_ann_vector::L2U8Distance`]) computes
//! `Σ(a-b)²` directly with `vabdq_u8 → vmull_u8 → vpadalq_u16`
//! — about **14 SIMD ops per 32-byte chunk** (2 abd + 4 widening mul
//! + 4 pairwise-add + 2 loads × 2 sides). Apple Silicon's `sdot`
//! instruction does 16 i8×i8 multiplies + a 4-way partial sum into
//! `int32x4_t` in **one** op — **2 sdot ops per 32-byte chunk** is all
//! the kernel does for inner-product compute.
//!
//! The L2-kernel-trick identity lets us substitute the IP path for L2:
//!
//! ```text
//!   ‖q - x‖²  =  ‖q‖² + ‖x‖² - 2·⟨q,x⟩
//! ```
//!
//! - `‖x‖²` is constant per base vertex → precompute once at build time,
//!   store as a parallel `i32` slab.
//! - `‖q‖²` is constant per query → compute once at search setup.
//! - `⟨q,x⟩` runs through `IpI8Distance` (sdot) per hop, ~2.3× cheaper
//!   per chunk than the direct-L2 kernel.
//!
//! Per-vertex overhead in the sink is **3 scalar f32 ops** (one
//! lookup, one add, one fma) — completely negligible vs the kernel
//! savings.
//!
//! ## Quantization
//!
//! Same affine slope as `L2U8`'s [`QuantParamsL2::from_range`]
//! (maps f32 input range → 256 levels), but the output is shifted to
//! the **signed-i8 range** `[-128, 127]` so `sdot` can consume it
//! directly. L2 distance is invariant under uniform translation, so
//! the i8 representation preserves the same precision as the existing
//! u8 sidecar — same per-dim quantization error, same rank fidelity.
//!
//! ## Storage layout
//!
//! - `data: AlignedBoxWithSlice<i8>` — `num_vertices × STRIDE` bytes,
//!   32-byte aligned for `ldp q,q` + sdot register loads.
//! - `norms_sq: AlignedBoxWithSlice<i32>` — `num_vertices × 4` bytes,
//!   `‖x_i8‖² = Σ x_i8²` per vertex. `i32` fits up to N=131K dims
//!   at max i8 squared (`127² = 16129`), well past any production
//!   workload.
//!
//! ## Sidecar file format (`.qdsl2kt`, magic `QDKT`)
//!
//! ```text
//! [u32 magic = QDKT][u32 version = 1]
//! [u32 num_vertices][u32 stride]
//! [u32 dim = N][u32 _pad]
//! [f32 slope][i32 offset]
//! [num_vertices × STRIDE bytes — i8 base]
//! [num_vertices × 4 bytes — i32 norms_sq]
//! ```

use super::quantized_dataset::{QuantizedDataset, QuantizedDatasetShape, ResidentMemory};
use super::rows::F32Rows;
use super::validation::{self, CheckedRows};
use orion_ann_diskann::common::AlignedBoxWithSlice;
use rayon::prelude::*;

use super::quantized_spec::QuantParamsL2;
use crate::Result;

/// Disk-format magic for the L2-kernel-trick sidecar (`.qdsl2kt`).
pub const L2_KT_MAGIC: u32 = 0x5144_4B54; // "QDKT"

/// Per-vertex stride in i8 elements. Rounds `N` up to a 32-element
/// boundary so each vertex starts on a 32-byte boundary (one `ldp
/// q,q` load) and the trailing `STRIDE - N` slots are zero-padded.
#[inline]
pub const fn l2_kt_stride(n: usize) -> usize {
    let bytes = l2_kt_stride_disk(n);
    let block =
        <orion_ann_vector::IpI8Distance as orion_ann_vector::DistanceKernelBackend>::CHUNK_BYTES;
    bytes.div_ceil(block) * block
}

pub const fn l2_kt_stride_disk(n: usize) -> usize {
    (n + 31) & !31
}

/// L2-kernel-trick sidecar: i8 base + per-vertex squared L2 norms.
pub struct L2KTDataset {
    pub(crate) dimension: usize,
    pub(crate) stride: usize,
    /// i8 quantized base, `num_vertices × STRIDE` bytes, 32-B aligned.
    pub(crate) data: AlignedBoxWithSlice<i8>,
    /// Per-vertex `‖x_i8‖²` in i32. Read once per candidate by the
    /// distance-reconstruction sink — fast scalar lookup off the hot
    /// SIMD path.
    pub(crate) norms_sq: AlignedBoxWithSlice<i32>,
    /// Number of vertices encoded.
    pub(crate) num_vertices: usize,
    /// f32 → i8 quantization params. Same affine `slope` as the
    /// L2-u8 sidecar; the i8 output is the u8 form minus 128.
    pub(crate) params: QuantParamsL2,
}

impl L2KTDataset {
    /// Per-vertex stride in i8 elements (= bytes). 32-element aligned.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Build the sidecar from an f32 base. Quantization params are
    /// derived from the f32 base's `(min, max)` range — same shape as
    /// `L2U8::build_params`. Per-vertex `‖x_i8‖²` is computed in the
    /// same parallel pass, no second sweep.
    /// Convenience constructor for already validated data.
    ///
    /// # Panics
    /// Panics on invalid input or allocation failure. Use [`Self::try_build_from`]
    /// to propagate a structured error at application boundaries.
    pub fn build_from(dataset: &impl F32Rows) -> Self {
        Self::try_build_from(dataset).expect("invalid sidecar input")
    }

    /// Construct a sidecar after validating shape, arithmetic and finite input.
    ///
    /// # Errors
    /// Returns an error for invalid rows, unsupported dimensions, invalid fitted
    /// parameters, or allocation failure. Logical rows exclude SIMD padding.
    pub fn try_build_from(dataset: &impl F32Rows) -> Result<Self> {
        let checked = CheckedRows::new(dataset, i32::MAX as usize / (128 * 128))?;
        let dataset = &checked;
        let dim = dataset.dimension();
        assert!(
            dim > 0
                && dim <= <orion_ann_vector::IpI8Distance as orion_ann_vector::DistanceKernelBackend>::MAX_COORDINATES
                && dim <= i32::MAX as usize / (128 * 128)
        );
        assert!(!dataset.is_empty());
        let num_vertices = dataset.len();
        let stride = l2_kt_stride(dim);

        // 1. Derive quantization params from the f32 base. Same scheme
        //    as `L2U8::build_params` so the i8 representation matches
        //    `u8 - 128` element-for-element.

        let (min_val, max_val) = dataset.range();
        let params = QuantParamsL2::from_range(min_val, max_val);
        validation::positive_scale(params.slope)?;

        // 2. Allocate the i8 base slab + norms_sq companion.
        let mut data =
            AlignedBoxWithSlice::<i8>::new(validation::product(num_vertices, stride)?, 32)?;
        let mut norms_sq = AlignedBoxWithSlice::<i32>::new(num_vertices, 32)?;

        // 3. Parallel quantize + per-vertex norm in a single pass.
        let data_slice = data.as_mut_slice();
        let norms_slice = norms_sq.as_mut_slice();
        data_slice
            .par_chunks_mut(stride)
            .zip(norms_slice.par_iter_mut())
            .enumerate()
            .for_each(|(vid, (out, norm_slot))| {
                let f32_v = dataset.row(vid);
                let mut sum: i32 = 0;
                for j in 0..dim {
                    let q = quantize_to_i8(&params, f32_v[j]);
                    out[j] = q;
                    sum += (q as i32) * (q as i32);
                }
                // out[dim..STRIDE] is already zero (alloc_zeroed).
                *norm_slot = sum;
            });

        Ok(Self {
            dimension: dim,
            stride,
            data,
            norms_sq,
            num_vertices,
            params,
        })
    }

    /// Quantize a query into a `[i8; N]` (short form, no padding).
    pub fn quantize_query(&self, query: &[f32]) -> Vec<i8> {
        assert_eq!(query.len(), self.dimension);
        query
            .iter()
            .map(|&v| quantize_to_i8(&self.params, v))
            .collect()
    }

    /// Quantize a query padded to `STRIDE` elements (trailing zeros).
    /// Matches the layout of base vertices so the streaming kernel
    /// can read full `STRIDE`-byte windows per vertex.
    pub fn quantize_query_padded(&self, query: &[f32]) -> Vec<i8> {
        assert_eq!(query.len(), self.dimension);
        let dim = self.dimension;
        let mut out = vec![0i8; self.stride];
        for i in 0..dim {
            out[i] = quantize_to_i8(&self.params, query[i]);
        }
        out
    }

    /// `‖q_i8‖²` computed by scalar accumulation over the first `N`
    /// elements of the (padded) query buffer. Cost is `O(N)` — paid
    /// once per query, sub-µs at production dims.
    #[inline]
    pub fn query_norm_sq(&self, q_query_padded: &[i8]) -> i32 {
        let dim = self.dimension;
        let mut sum: i32 = 0;
        for j in 0..dim {
            let v = q_query_padded[j] as i32;
            sum += v * v;
        }
        sum
    }

    /// Sidecar header (40 B):
    ///   [u32 magic = L2_KT_MAGIC][u32 version = 1]
    ///   [u32 num_vertices][u32 stride]
    ///   [u32 dim = N][u32 _pad]
    ///   [f32 slope][i32 offset]
    /// Body:
    ///   [num_vertices × STRIDE i8 base]
    ///   [num_vertices × 4 i32 norms_sq]
    pub fn encode(&self, w: &mut dyn std::io::Write) -> Result<()> {
        let dim = self.dimension;
        let mut hdr = [0u8; 40];
        hdr[0..4].copy_from_slice(&L2_KT_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(l2_kt_stride_disk(dim) as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(dim as u32).to_le_bytes());
        // hdr[20..24]: pad
        hdr[24..28].copy_from_slice(&self.params.slope.to_le_bytes());
        hdr[28..32].copy_from_slice(&self.params.offset.to_le_bytes());
        // hdr[32..40]: reserved for future
        w.write_all(&hdr)?;

        // i8 base — reinterpret as u8 bytes for the byte writer.
        super::rows::write_rows(w, self.data.as_slice(), self.stride, l2_kt_stride_disk(dim))?;

        // i32 norms_sq.
        let norms_bytes = bytemuck::cast_slice(self.norms_sq.as_slice());
        w.write_all(norms_bytes)?;
        Ok(())
    }

    pub fn decode(r: &mut dyn std::io::Read, byte_len: u64) -> Result<Self> {
        let mut hdr = [0u8; 40];
        r.read_exact(&mut hdr)?;
        if u32::from_le_bytes(hdr[4..8].try_into().unwrap()) != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported sidecar version",
            )
            .into());
        }
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != L2_KT_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = QDKT)",
                    L2_KT_MAGIC
                ),
            )
            .into());
        }
        let num_vertices = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        validation::shape(num_vertices, dim, i32::MAX as usize / (128 * 128))?;
        let stride = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if stride != l2_kt_stride_disk(dim) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "stride mismatch: file={stride} expected={}",
                    l2_kt_stride(dim)
                ),
            )
            .into());
        }
        if dim == 0
            || dim
                > <orion_ann_vector::IpI8Distance as orion_ann_vector::DistanceKernelBackend>::MAX_COORDINATES
            || dim > i32::MAX as usize / (128 * 128)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "zero sidecar dimension",
            )
            .into());
        }
        let slope = f32::from_le_bytes(hdr[24..28].try_into().unwrap());
        let offset = i32::from_le_bytes(hdr[28..32].try_into().unwrap());
        let params = QuantParamsL2 { slope, offset };

        validation::positive_scale(slope)?;
        validation::file_length(byte_len, 40, &[(num_vertices, stride), (num_vertices, 4)])?;

        // i8 base.
        let disk_stride = stride;
        let stride = l2_kt_stride(dim);
        let data = super::rows::read_rows::<i8>(r, num_vertices, stride, disk_stride)?;

        // i32 norms_sq.
        let mut norms_sq = AlignedBoxWithSlice::<i32>::new(num_vertices, 32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        let norms_bytes = bytemuck::cast_slice_mut(norms_sq.as_mut_slice());
        r.read_exact(norms_bytes)?;

        if norms_sq
            .as_slice()
            .iter()
            .any(|&n| n < 0 || n as usize > dim * 128 * 128)
        {
            return Err(crate::Error::invalid_data(
                "L2KT norm exceeds the quantized row bound",
            ));
        }
        Ok(Self {
            dimension: dim,
            stride,
            data,
            norms_sq,
            num_vertices,
            params,
        })
    }
}

/// f32 → i8 via the same affine map as `L2U8`, shifted by -128 so
/// the output lives in i8 range. Equivalent to
/// `L2U8::quantize_scalar(v) - 128` but skips the intermediate u8
/// clamp by directly clamping to `[-128, 127]`.
#[inline]
fn quantize_to_i8(p: &QuantParamsL2, v: f32) -> i8 {
    let x = ((v * p.slope).round() as i64)
        .saturating_sub(p.offset as i64)
        .saturating_sub(128);
    x.clamp(-128, 127) as i8
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for L2KTDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2KTDataset")
            .field("dimension", &self.dimension)
            .field("num_vertices", &self.num_vertices)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_diskann::model::InmemDataset;

    #[test]
    fn stride_matches_alignment() {
        // N=128 (SIFT): already a 32-element multiple.
        assert_eq!(l2_kt_stride(128), 128);
        // N=100 (glove): rounds up to 128 (= 4 × 32 elem boundary).
        assert_eq!(l2_kt_stride(100), 128);
        // N=960 (GIST): already a 32-element multiple.
        assert_eq!(l2_kt_stride(960), 960);
        // N=33: rounds to 64.
        assert_eq!(l2_kt_stride(33), 64);
    }

    #[test]
    fn quantize_matches_u8_minus_128() {
        let p = QuantParamsL2::from_range(0.0, 255.0);
        for v in [0.0_f32, 1.5, 64.0, 127.0, 128.0, 200.0, 255.0] {
            let u = p.quantize_scalar(v) as i32;
            let i = quantize_to_i8(&p, v) as i32;
            assert_eq!(u - 128, i, "v={v}, u8={u}, i8={i}");
        }
    }

    #[test]
    fn l2_invariant_under_shift() {
        // ‖a - b‖² is invariant under uniform translation; verify the
        // i8 sidecar gives the same squared distance as its u8 twin.
        let p = QuantParamsL2::from_range(0.0, 255.0);
        let a = [10.0_f32, 50.0, 200.0, 5.0];
        let b = [80.0_f32, 10.0, 100.0, 220.0];
        let a_u8: Vec<i32> = a.iter().map(|&v| p.quantize_scalar(v) as i32).collect();
        let b_u8: Vec<i32> = b.iter().map(|&v| p.quantize_scalar(v) as i32).collect();
        let a_i8: Vec<i32> = a.iter().map(|&v| quantize_to_i8(&p, v) as i32).collect();
        let b_i8: Vec<i32> = b.iter().map(|&v| quantize_to_i8(&p, v) as i32).collect();
        let l2_u8: i32 = a_u8
            .iter()
            .zip(b_u8.iter())
            .map(|(a, b)| (a - b).pow(2))
            .sum();
        let l2_i8: i32 = a_i8
            .iter()
            .zip(b_i8.iter())
            .map(|(a, b)| (a - b).pow(2))
            .sum();
        assert_eq!(l2_u8, l2_i8);
    }

    #[test]
    fn kernel_trick_identity() {
        // d²(q,x) = ‖q‖² + ‖x‖² - 2·⟨q,x⟩ should hold exactly for i8
        // values when accumulated in i32.
        let q: [i32; 4] = [12, -34, 56, -78];
        let x: [i32; 4] = [-20, 41, -10, 33];
        let direct: i32 = q.iter().zip(x.iter()).map(|(a, b)| (a - b).pow(2)).sum();
        let q_sq: i32 = q.iter().map(|v| v * v).sum();
        let x_sq: i32 = x.iter().map(|v| v * v).sum();
        let ip: i32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let kt = q_sq + x_sq - 2 * ip;
        assert_eq!(direct, kt);
    }

    #[test]
    fn build_then_quantize_roundtrip() {
        // Tiny synthetic dataset: build the sidecar then verify the
        // per-vertex norm matches a scalar recomputation.
        const D: usize = 8;
        let num = 5;
        let flat: Vec<f32> = (0..num * D).map(|i| (i % 200) as f32).collect();
        let mut ds = InmemDataset::<f32, D>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let kt = L2KTDataset::build_from(&ds);

        // Per-vertex norms_sq must match a direct compute from the
        // stored i8 data.
        let stride = kt.stride;
        for vid in 0..num {
            let off = vid * stride;
            let mut expected: i32 = 0;
            for j in 0..D {
                let v = kt.data.as_slice()[off + j] as i32;
                expected += v * v;
            }
            assert_eq!(kt.norms_sq.as_slice()[vid], expected, "vid={vid}");
        }
    }
}

impl L2KTDataset {
    /// Read-only access to the sidecar dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Read-only access to the sidecar data.
    #[inline]
    pub fn data(&self) -> &[i8] {
        self.data.as_slice()
    }
    /// Read-only access to the sidecar norms sq.
    #[inline]
    pub fn norms_sq(&self) -> &[i32] {
        self.norms_sq.as_slice()
    }
    /// Read-only access to the sidecar num vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_vertices
    }
    /// Whether this sidecar contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_vertices == 0
    }
    /// Read-only access to the sidecar params.
    #[inline]
    pub fn params(&self) -> &QuantParamsL2 {
        &self.params
    }
}

impl super::quantized_dataset::sealed::Sealed for L2KTDataset {}
impl QuantizedDataset for L2KTDataset {
    type BuildOptions = ();
    fn default_build_options() -> Self::BuildOptions {
        ()
    }
    fn build(base: &impl F32Rows, _options: &Self::BuildOptions) -> Result<Self> {
        Self::estimate_resident_memory(
            QuantizedDatasetShape {
                rows: base.len(),
                dimension: base.dimension(),
            },
            _options,
        )?;
        Self::try_build_from(base)
    }
    fn estimate_resident_memory(
        input_shape: QuantizedDatasetShape,
        _options: &Self::BuildOptions,
    ) -> Result<ResidentMemory> {
        input_shape.validate()?;
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpI8Distance>(
            input_shape.dimension,
        )?;
        ResidentMemory::new(
            validation::product(input_shape.rows, l2_kt_stride(input_shape.dimension))?,
            validation::product(input_shape.rows, 4)?,
            0,
        )
    }
    fn resident_memory(&self) -> ResidentMemory {
        ResidentMemory::new(
            self.data.len(),
            std::mem::size_of_val(self.norms_sq.as_slice()),
            0,
        )
        .expect("validated resident allocation sizes")
    }
}
