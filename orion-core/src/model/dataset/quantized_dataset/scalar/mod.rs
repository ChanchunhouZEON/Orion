/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! memory: N rows of [D codes of Q::Storage | neutral zero padding] + Q::Params
//! file:   [24 B header: magic/version/N/D/parameters][N * D packed codes]
//! ```
//!
//! The scalar specification chooses code width and metric. Headers are
//! little-endian; parameter/code payloads use their POD representation. File
//! rows omit SIMD padding, which is rebuilt for the selected kernel at load.
//! `ArrayView` provides the same row layout for owned and private-mapped data.

//! Padded scalar-quantized storage. Codec rules live in `quantized_spec`;
//! the common dataset contract includes projection and kernel-trick stores too.
use crate::model::dataset::quantized_dataset::{
    QuantizedDataset, QuantizedDatasetShape, ResidentMemory,
};
pub mod spec;
use crate::Result;
use crate::model::dataset::rows::Rows;
use crate::model::dataset::validation::{self, CheckedRows};
use orion_ann_platform::region::{ArrayView, Placement, RegionOptions};
use orion_ann_vector::DistanceKernelBackend;
use spec::*;

// ─── Generic dataset ──────────────────────────────────────────────────────

/// Quantized base store. `Q` selects the storage + metric (one of
/// [`L2U8`], [`L2U16`], [`MipsI8`], [`MipsI16`]). Logical dimensions
/// and SIMD padding are validated at construction and cannot be changed later.
///
/// ```compile_fail
/// use orion_ann_core::model::{ScalarQuantizedDataset, L2U8};
/// fn invalidate(dataset: &mut ScalarQuantizedDataset<L2U8>) {
///     dataset.n = usize::MAX;
/// }
/// ```
/// ```compile_fail
/// use orion_ann_core::model::{ScalarQuantizedDataset, L2U8};
/// fn mutate_codes(dataset: &mut ScalarQuantizedDataset<L2U8>) {
///     dataset.data()[0] = 0;
/// }
/// ```
pub struct ScalarQuantizedDataset<Q: QuantSpec> {
    pub(crate) data: ArrayView<Q::Storage>,
    pub(crate) params: Q::Params,
    pub(crate) n: usize,
    pub(crate) dimension: usize,
    pub(crate) stride: usize,
}

impl<Q: QuantSpec> ScalarQuantizedDataset<Q> {
    /// Construct a quantized base with each row padded to the selected kernel.
    ///
    /// Convenience constructor for already validated data.
    ///
    /// # Panics
    /// Panics on invalid input or allocation failure. Use [`Self::try_from_f32_dataset`]
    /// to propagate a structured error at application boundaries.
    pub fn from_f32_dataset(ds: &impl Rows) -> Self {
        Self::try_from_f32_dataset(ds).expect("invalid sidecar input")
    }

    /// Construct a sidecar after validating shape, arithmetic and finite input.
    ///
    /// # Errors
    /// Returns an error for invalid rows, unsupported dimensions, invalid fitted
    /// parameters, or allocation failure. Logical rows exclude SIMD padding.
    pub fn try_from_f32_dataset(ds: &impl Rows) -> Result<Self> {
        Self::build_with_storage(ds, &RegionOptions::default())
    }

    /// Build directly into the selected backing, without a full heap staging copy.
    /// # Errors
    /// Reports invalid input/layout, allocation and private-file I/O failures.
    pub fn build_with_storage(ds: &impl Rows, storage: &RegionOptions) -> Result<Self> {
        let checked = CheckedRows::new(ds, Q::QuantKernel::MAX_COORDINATES)?;
        let ds = &checked;
        let dimension = ds.dimension();
        let n = ds.len();
        assert!(dimension > 0 && n > 0);
        assert!(dimension <= Q::QuantKernel::MAX_COORDINATES);
        let layout = orion_ann_vector::PaddedLayout::for_kernel::<Q::QuantKernel>(dimension)
            .expect("valid quantized row layout");
        let stride = layout.stride_elements();
        // Padding is not a sample: only logical values may fit the quantizer.
        let (min, max) = ds.range();
        let params = Q::build_params(&[min, max]);
        Q::validate_params(&params)?;
        use rayon::prelude::*;
        let data = ArrayView::build_rows::<crate::Error>(
            n,
            stride,
            layout.alignment(),
            storage,
            |start, rows| {
                rows.par_chunks_mut(stride).enumerate().for_each_init(
                    Vec::new,
                    |row_scratch, (id, out)| {
                        for (dst, &value) in out.iter_mut().zip(ds.row_f32(start + id, row_scratch))
                        {
                            *dst = Q::quantize_scalar(&params, value);
                        }
                    },
                );
                Ok(())
            },
        )?;
        Ok(Self {
            data,
            params,
            n,
            dimension,
            stride,
        })
    }

    /// # Safety
    /// `id` must be less than [`Self::len`]. The immutable, validated layout
    /// guarantees the returned logical row lies within the owned allocation.
    #[inline]
    pub unsafe fn get_vertex_unchecked(&self, id: u32) -> &[Q::Storage] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr().add(id as usize * self.stride),
                self.dimension,
            )
        }
    }

    /// Borrow one logical row; SIMD padding is excluded.
    #[inline]
    pub fn row(&self, id: u32) -> Option<&[Q::Storage]> {
        if id as usize >= self.n {
            return None;
        }
        let start = id as usize * self.stride;
        Some(&self.data.as_slice()[start..start + self.dimension])
    }

    /// Compute a quantized distance with checked ID and query length.
    ///
    /// # Errors
    /// Rejects an out-of-range ID or a query with neither logical nor padded length.
    pub fn distance(&self, id: u32, query: &[Q::Storage]) -> Result<f32> {
        if id as usize >= self.n {
            return Err(
                crate::Error::invalid_argument("sidecar row ID out of bounds")
                    .with_detail("id", id),
            );
        }
        if query.len() != self.dimension && query.len() != self.stride {
            return Err(crate::Error::dimension(
                "quantized query",
                self.dimension,
                query.len(),
            ));
        }
        // SAFETY: ID and query storage were checked above; layout is immutable.
        Ok(unsafe { self.qdist(id, query) })
    }

    /// Quantize a finite query, optionally adding neutral SIMD padding.
    ///
    /// # Errors
    /// Returns a dimension error or rejects non-finite coordinates.
    pub fn try_quantize_query(&self, query: &[f32], padded: bool) -> Result<Vec<Q::Storage>> {
        if query.len() != self.dimension {
            return Err(crate::Error::dimension(
                "query",
                self.dimension,
                query.len(),
            ));
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(crate::Error::invalid_argument(
                "quantization requires finite query coordinates",
            ));
        }
        let mut output =
            vec![Q::Storage::default(); if padded { self.stride } else { self.dimension }];
        for (out, &value) in output.iter_mut().zip(query) {
            *out = Q::quantize_scalar(&self.params, value);
        }
        Ok(output)
    }

    /// Quantize only the logical coordinates. `qdist` accepts this packed
    /// form, while search sessions use the padded form to avoid per-hop copies.
    pub fn quantize_query(&self, query: &[f32]) -> Vec<Q::Storage> {
        self.try_quantize_query(query, false)
            .expect("invalid quantized query")
    }

    /// Quantize a query into a heap-allocated `Vec<Storage>` of size
    /// `self.stride` — the same per-vertex element span as the base
    /// buffer, with the trailing `STRIDE - N` slots zero-padded so
    /// the kernel sees the same byte footprint per vertex on both
    /// sides. This is the form [`orion_ann_vector::DistanceStream`] expects:
    /// it reads complete kernel blocks from
    /// both the base and the query, so any byte the base touches must
    /// also be present (and zero) in the query. Allocated once per
    /// `search<Q>` call, reused across every hop.
    pub fn quantize_query_padded(&self, query: &[f32]) -> Vec<Q::Storage> {
        self.try_quantize_query(query, true)
            .expect("invalid quantized query")
    }

    /// Quantized distance between vertex `id` and the query (already
    /// cast to f32, "smaller == closer"). Caller should compare against
    /// `f32_threshold * Q::distance_scale_sq(self.params) * Q_SLACK` to
    /// stay in the same comparison space.
    ///
    /// # Safety
    /// `id` must be less than [`Self::len`]. Query padding, when supplied,
    /// must be zero for the result to represent logical-coordinate distance.
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
            orion_ann_vector::DistanceStream::<Q::QuantKernel>::new(
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
    ///
    /// # Safety
    /// Every ID and query must satisfy [`Self::qdist`]'s contract.
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
        assert!((id as usize) < self.n, "sidecar row ID out of bounds");
        // SAFETY: the checked ID addresses an immutable, validated logical row.
        unsafe {
            let v = self.get_vertex_unchecked(id);
            orion_ann_vector::prefetch_vector(v);
        }
    }

    /// Header layout (24 B):
    ///   [u32 magic = Q::MAGIC]
    ///   [u32 version = 1]
    ///   [u32 n][u32 dim = N]
    ///   [Q::Params packed into 8 B]   ← spec-specific
    /// Body: packed `n × N` storage elements (no STRIDE padding).
    pub fn encode(&self, w: &mut dyn std::io::Write) -> Result<()> {
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
        for row in self.data.as_slice().chunks_exact(self.stride) {
            w.write_all(bytemuck::cast_slice(&row[..self.dimension]))?;
        }
        Ok(())
    }

    pub fn decode(r: &mut dyn std::io::Read, byte_len: u64) -> Result<Self> {
        Self::decode_with_storage(r, byte_len, &RegionOptions::default())
    }

    /// Decode packed cache rows into aligned owned or private mapped storage.
    /// # Errors
    /// Validates shape, parameters and payload length before allocating.
    pub fn decode_with_storage(
        r: &mut dyn std::io::Read,
        byte_len: u64,
        storage: &RegionOptions,
    ) -> Result<Self> {
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
        if dim == 0 || n == 0 || dim > Q::QuantKernel::MAX_COORDINATES {
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
        if expected != Some(byte_len) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "quantized payload length does not match its shape",
            )
            .into());
        }
        let layout = orion_ann_vector::PaddedLayout::for_kernel::<Q::QuantKernel>(dim)
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
        Q::validate_params(&params)?;
        let data = ArrayView::build_rows::<crate::Error>(
            n,
            stride,
            layout.alignment(),
            storage,
            |_, out| {
                if stride == dim {
                    r.read_exact(bytemuck::cast_slice_mut(out))?;
                } else {
                    for row in out.chunks_exact_mut(stride) {
                        r.read_exact(bytemuck::cast_slice_mut(&mut row[..dim]))?;
                    }
                }
                Ok(())
            },
        )?;
        Ok(Self {
            data,
            params,
            n,
            dimension: dim,
            stride,
        })
    }
}

impl<Q: QuantSpec> std::fmt::Debug for ScalarQuantizedDataset<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarQuantizedDataset")
            .field("quantizer", &std::any::type_name::<Q>())
            .field("points", &self.n)
            .field("dimension", &self.dimension)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl<Q: QuantSpec> ScalarQuantizedDataset<Q> {
    /// Placement of the immutable quantized payload.
    #[inline]
    pub fn placement(&self) -> Placement {
        self.data.placement()
    }

    /// Read-only access to the sidecar data.
    #[inline]
    pub fn data(&self) -> &[Q::Storage] {
        self.data.as_slice()
    }
    /// Read-only access to the sidecar params.
    #[inline]
    pub fn params(&self) -> &Q::Params {
        &self.params
    }
    /// Read-only access to the sidecar n.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }
    /// Whether this sidecar contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// Read-only access to the sidecar dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Read-only access to the sidecar stride.
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }
}

impl<Q: QuantSpec> crate::model::dataset::quantized_dataset::sealed::Sealed
    for ScalarQuantizedDataset<Q>
{
}
impl<Q: QuantSpec> QuantizedDataset for ScalarQuantizedDataset<Q> {
    type BuildOptions = ();
    fn default_build_options() -> Self::BuildOptions {}
    fn build(base: &impl Rows, _options: &Self::BuildOptions) -> Result<Self> {
        Self::estimate_resident_memory(
            QuantizedDatasetShape {
                rows: base.len(),
                dimension: base.dimension(),
            },
            _options,
        )?;
        Self::try_from_f32_dataset(base)
    }
    fn estimate_resident_memory(
        input_shape: QuantizedDatasetShape,
        _options: &Self::BuildOptions,
    ) -> Result<ResidentMemory> {
        input_shape.validate()?;
        let layout =
            orion_ann_vector::PaddedLayout::for_kernel::<Q::QuantKernel>(input_shape.dimension)?;
        ResidentMemory::new(layout.allocation_bytes(input_shape.rows)?, 0, 0)
    }
    fn resident_memory(&self) -> ResidentMemory {
        ResidentMemory::new(std::mem::size_of_val(self.data.as_slice()), 0, 0)
            .expect("validated resident allocation sizes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use orion_ann_diskann::model::InmemDataset;

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

    // ─── ScalarQuantizedDataset<L2U8> ──────────────────────────────────────

    #[test]
    fn build_l2u8_dataset() {
        let ds = make_ds::<8>(16);
        let qds = ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&ds);
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
        let qds = ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&ds);
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
        let qds = ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&ds);
        qds.encode(&mut std::fs::File::create(&tmp).unwrap())
            .unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        let loaded =
            ScalarQuantizedDataset::<L2U8>::decode(&mut bytes.as_slice(), bytes.len() as u64)
                .unwrap();
        assert_eq!(loaded.n, qds.n);
        assert!((loaded.params.slope - qds.params.slope).abs() < 1e-6);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn build_mips_i8_dataset() {
        let ds = make_ds::<8>(4);
        let qds = ScalarQuantizedDataset::<MipsI8>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 4);
        assert_eq!((qds.stride) % 32, 0);
    }

    #[test]
    fn build_mips_i16_dataset() {
        let ds = make_ds::<8>(4);
        let qds = ScalarQuantizedDataset::<MipsI16>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 4);
        assert_eq!((qds.stride * 2) % 32, 0);
    }

    #[test]
    fn build_l2_u16_dataset() {
        let ds = make_ds::<8>(8);
        let qds = ScalarQuantizedDataset::<L2U16>::from_f32_dataset(&ds);
        assert_eq!(qds.n, 8);
    }
}
