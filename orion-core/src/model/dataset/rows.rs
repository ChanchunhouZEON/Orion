/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! backing row: [D logical coordinates | optional physical padding]
//! Rows view:  [D f32 coordinates], borrowed or converted into reusable scratch
//! ```
//!
//! `Rows` owns no storage and defines no file format. Row adapters hide stride
//! and element width from sidecar fitting. Packed-file helpers below write
//! codec-defined row widths and reconstruct additional in-memory zero padding.

//! Logical rows are independent of backing element width and physical stride.
//! Sidecar mathematics consumes f32 views: f32 rows are borrowed, other element
//! types convert into reusable worker scratch. No full-base conversion is needed.
use orion_ann_diskann::model::{InmemDataset, PaddedDataset};

/// Logical coordinates used to fit a sidecar, independent of storage stride.
///
/// `row_f32` borrows f32 rows or converts into caller-owned scratch. Numeric
/// conversion may round wider elements; it does not enable a distance kernel
/// for that element type. Implementations must return `dimension()` values.
/// Dataset builders validate this contract before constructing a sidecar.
pub trait Rows: Sync {
    /// Returns meaningful coordinates per logical row, excluding physical padding.
    fn dimension(&self) -> usize;

    /// Returns the number of addressable logical rows.
    fn len(&self) -> usize;

    /// Borrows a floating row or converts it into the supplied reusable scratch.
    ///
    /// The returned slice must have exactly `dimension()` coordinates. Callers pass
    /// a valid row ID; implementations may panic for out-of-range IDs.
    fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32];

    /// Replaces the output with one logical row, preserving reusable output capacity.
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        let mut scratch = Vec::new();
        let row = self.row_f32(id, &mut scratch);
        output.clear();
        output.extend_from_slice(row);
    }

    /// Reports whether this view contains no logical rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scans finite logical coordinates to determine the fitting range.
    ///
    /// # Panics
    ///
    /// The default implementation panics on non-finite coordinates.
    fn range(&self) -> (f32, f32) {
        let mut scratch = Vec::new();
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for id in 0..self.len() {
            for &value in self.row_f32(id, &mut scratch) {
                assert!(
                    value.is_finite(),
                    "quantization requires finite logical coordinates"
                );
                min = min.min(value);
                max = max.max(value);
            }
        }
        (min, max)
    }
}
// Storage does not depend on the set of distance kernels. These adapters are
// numerical input views for sidecar fitting, not search-backend selections.
impl Rows for PaddedDataset<f32> {
    /// Reports logical coordinates per row, excluding storage padding.
    fn dimension(&self) -> usize {
        self.layout().logical_dim()
    }

    /// Reports logical rows available to sidecar fitting.
    fn len(&self) -> usize {
        self.len()
    }

    /// Borrows a logical f32 row or converts it into reusable caller scratch.
    fn row_f32<'a>(&'a self, id: usize, _: &'a mut Vec<f32>) -> &'a [f32] {
        self.row(id).expect("row ID out of bounds")
    }

    /// Replaces caller scratch with one logical row in f32 representation.
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        output.clear();
        output.extend_from_slice(self.row(id).expect("row ID out of bounds"));
    }
}

// Conversion is numeric (never a byte reinterpretation). Wide integers and
// f64 may round to f32; builders validate the resulting finite coordinates.
macro_rules! numeric_rows {
    ($($element:ty),+ $(,)?) => {$(
        impl Rows for PaddedDataset<$element> {
            fn dimension(&self) -> usize { self.layout().logical_dim() }
            fn len(&self) -> usize { self.len() }
            fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
                self.copy_row_f32(id, scratch);
                scratch
            }
            fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
                output.clear();
                output.extend(self.row(id).expect("row ID out of bounds").iter().map(|&v| v as f32));
            }
        }
    )+};
}
numeric_rows!(u8, i8, u16, i16, u32, i32, u64, i64, f64);

impl<const N: usize> Rows for InmemDataset<f32, N> {
    /// Reports logical coordinates per row, excluding storage padding.
    fn dimension(&self) -> usize {
        N
    }

    /// Reports logical rows available to sidecar fitting.
    fn len(&self) -> usize {
        self.num_active_pts
    }

    /// Borrows a logical f32 row or converts it into reusable caller scratch.
    fn row_f32<'a>(&'a self, id: usize, _: &'a mut Vec<f32>) -> &'a [f32] {
        &self.data[id * N..(id + 1) * N]
    }

    /// Replaces caller scratch with one logical row in f32 representation.
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        output.clear();
        output.extend_from_slice(&self.data[id * N..(id + 1) * N]);
    }
}

/// Input transform for fitted sidecars; never changes the immutable base.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RowTransform {
    #[default]
    Identity,
    UnitL2,
}

/// Borrowed fitting view that transforms logical coordinates without mutating the base.
pub(crate) struct TransformedRows<'a, D> {
    /// Original rows whose storage and padding remain unchanged.
    pub base: &'a D,

    /// Per-row transform applied only to the fitting view.
    pub transform: RowTransform,
}
impl<D: Rows> Rows for TransformedRows<'_, D> {
    /// Reports logical coordinates per row, excluding storage padding.
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    /// Reports logical rows available to sidecar fitting.
    fn len(&self) -> usize {
        self.base.len()
    }

    /// Borrows a logical f32 row or converts it into reusable caller scratch.
    fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
        if self.transform == RowTransform::Identity {
            return self.base.row_f32(id, scratch);
        }
        self.copy_row_f32(id, scratch);
        scratch
    }

    /// Replaces caller scratch with one logical row in f32 representation.
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        self.base.copy_row_f32(id, output);
        if self.transform == RowTransform::UnitL2 {
            let norm = output
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for v in output.iter_mut() {
                    *v = (f64::from(*v) / norm) as f32;
                }
            }
        }
    }
}

/// In-memory code rows contain complete distance-kernel blocks. Disk rows keep
/// the historical format; readers reconstruct zero padding for this machine.
pub(crate) fn write_rows<T: bytemuck::Pod>(
    writer: &mut dyn std::io::Write,
    data: &[T],
    memory_stride: usize,
    disk_stride: usize,
) -> std::io::Result<()> {
    if memory_stride == disk_stride {
        return writer.write_all(bytemuck::cast_slice(data));
    }
    for row in data.chunks_exact(memory_stride) {
        writer.write_all(bytemuck::cast_slice(&row[..disk_stride]))?;
    }
    Ok(())
}

/// Decodes packed rows into selected backing and reconstructs zero-filled memory padding.
pub(crate) fn read_rows_with_storage<T: bytemuck::Pod>(
    reader: &mut dyn std::io::Read,
    rows: usize,
    memory_stride: usize,
    disk_stride: usize,
    storage: &orion_ann_platform::region::RegionOptions,
) -> orion_ann_diskann::common::ANNResult<orion_ann_diskann::common::AlignedBoxWithSlice<T>> {
    use std::io::{Error, ErrorKind};
    let count = rows
        .checked_mul(memory_stride)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "sidecar allocation overflow"))?;
    if rows == 0 || disk_stride == 0 || disk_stride > memory_stride {
        return Err(Error::new(ErrorKind::InvalidData, "invalid sidecar row layout").into());
    }
    let mut data =
        orion_ann_diskann::common::AlignedBoxWithSlice::<T>::new_with_storage(count, 64, storage)?;
    let row_bytes = memory_stride
        .checked_mul(size_of::<T>())
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "invalid sidecar row size"))?;
    let chunk_rows = (orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES / row_bytes).max(1);
    for start in (0..rows).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(rows);
        let block = &mut data[start * memory_stride..end * memory_stride];
        if memory_stride == disk_stride {
            orion_ann_platform::buffered_io::read_exact_in_blocks(
                reader,
                bytemuck::cast_slice_mut(block),
            )?;
        } else {
            for row in block.chunks_exact_mut(memory_stride) {
                reader.read_exact(bytemuck::cast_slice_mut(&mut row[..disk_stride]))?;
            }
        }
        if storage.reclaim_after_write {
            data.flush_and_reclaim(start * row_bytes..end * row_bytes)?;
        }
    }
    Ok(data)
}

/// Full-precision stage factories for the two storage facades. Application
/// code is generic over storage; all new frontends instantiate padded rows.
pub trait CascadeData: Rows + crate::index::IndexData + Send {
    /// Exposes a native padded-f32 view when this storage representation provides one.
    fn padded_f32(&self) -> Option<&PaddedDataset<f32>> {
        None
    }

    /// Exposes a native padded-byte view when this storage representation provides one.
    fn padded_u8(&self) -> Option<&PaddedDataset<u8>> {
        None
    }

    /// Borrows the complete base allocation span for storage accounting and pinning.
    fn allocation_bytes(&self) -> &[u8];

    /// Creates the benchmark-only adaptive-sampling stage when this base type supports it.
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        epsilon: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>>;

    /// Creates an original-coordinate reranker for squared L2 or inner-product ordering.
    fn truth(
        &self,
        inner_product: bool,
    ) -> Box<dyn crate::search::cascade::stage::RerankStage + '_>;
}
impl CascadeData for PaddedDataset<f32> {
    /// Exposes the padded f32 base to exact stage dispatch.
    fn padded_f32(&self) -> Option<&PaddedDataset<f32>> {
        Some(self)
    }

    /// Borrows the underlying base payload for placement and residency bookkeeping.
    fn allocation_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }

    /// Opens the benchmark ADSampling stage when supported by this base representation.
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        epsilon: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>> {
        Ok(Box::new(
            crate::search::cascade::stage::admission::ads_f32::DynamicAdsF32Admission::new(
                self, epsilon,
            ),
        ))
    }

    /// Opens original-coordinate reranking with the requested L2 or IP distance semantics.
    fn truth(&self, ip: bool) -> Box<dyn crate::search::cascade::stage::RerankStage + '_> {
        use crate::search::cascade::stage::rerank::DynamicF32Rerank;
        if ip {
            Box::new(DynamicF32Rerank::<orion_ann_vector::IpF32Distance>::new(
                self,
            ))
        } else {
            Box::new(DynamicF32Rerank::<orion_ann_vector::L2F32Distance>::new(
                self,
            ))
        }
    }
}
impl<const N: usize> CascadeData for InmemDataset<f32, N> {
    /// Borrows the underlying base payload for placement and residency bookkeeping.
    fn allocation_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
    }

    /// Opens the benchmark ADSampling stage when supported by this base representation.
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        epsilon: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>> {
        Ok(Box::new(
            crate::search::cascade::stage::admission::AdsF32Admission::new(self, epsilon),
        ))
    }

    /// Opens original-coordinate reranking with the requested L2 or IP distance semantics.
    fn truth(&self, ip: bool) -> Box<dyn crate::search::cascade::stage::RerankStage + '_> {
        use crate::search::cascade::stage::rerank::{F32Rerank, IpF32Rerank};
        if ip {
            Box::new(IpF32Rerank::new(self))
        } else {
            Box::new(F32Rerank::new(self))
        }
    }
}

impl CascadeData for PaddedDataset<u8> {
    /// Exposes the padded byte base to exact stage dispatch.
    fn padded_u8(&self) -> Option<&PaddedDataset<u8>> {
        Some(self)
    }

    /// Borrows the underlying base payload for placement and residency bookkeeping.
    fn allocation_bytes(&self) -> &[u8] {
        self.as_slice()
    }

    /// Rejects ADSampling because this base stores native bytes rather than f32 rows.
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        _: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>> {
        Err(crate::Error::unsupported(
            "ADSampling requires f32 base storage",
        ))
    }

    /// Opens original-coordinate reranking with the requested L2 or IP distance semantics.
    fn truth(&self, ip: bool) -> Box<dyn crate::search::cascade::stage::RerankStage + '_> {
        let metric = if ip {
            crate::ScoreMetric::InnerProduct
        } else {
            crate::ScoreMetric::SquaredL2
        };
        Box::new(
            crate::model::dataset::quantized_dataset::exact::stage::ExactStage::u8(
                self, None, metric,
            )
            .expect("validated exact layout"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_vector::PaddedLayout;

    #[test]
    fn numeric_views_ignore_padding_and_reuse_conversion_scratch() {
        let layout = PaddedLayout::new::<u16>(3, 32).unwrap();
        let base =
            PaddedDataset::from_packed(&[1u16, 256, 65_535, 2, 400, 50_000], layout).unwrap();
        let mut scratch = Vec::with_capacity(3);
        let ptr = scratch.as_ptr();
        assert_eq!(base.row_f32(0, &mut scratch), &[1., 256., 65_535.]);
        assert_eq!(base.row_f32(1, &mut scratch), &[2., 400., 50_000.]);
        assert_eq!(scratch.as_ptr(), ptr);
        assert_eq!(base.range(), (1., 65_535.));
        assert_eq!(base.row(0).unwrap().len(), 3);
        assert_eq!(base.as_slice().len(), 2 * layout.stride_elements());
    }

    #[test]
    fn f32_rows_are_borrowed_without_conversion() {
        let base =
            PaddedDataset::from_packed(&[3.0f32, 4.0], PaddedLayout::new::<f32>(2, 32).unwrap())
                .unwrap();
        let mut scratch = Vec::new();
        assert_eq!(
            base.row_f32(0, &mut scratch).as_ptr(),
            base.row(0).unwrap().as_ptr()
        );
        assert_eq!(scratch.capacity(), 0);
    }

    #[test]
    fn storage_does_not_require_a_search_kernel() {
        // Even a compound POD coordinate is legal storage. No Rows conversion or
        // distance kernel is implicitly claimed for this element type.
        let values = [[1u32, 2], [3, 4], [5, 6]];
        let layout = PaddedLayout::new::<[u32; 2]>(3, 32).unwrap();
        let base = PaddedDataset::from_packed(&values, layout).unwrap();
        assert_eq!(base.row(0).unwrap(), values);
        assert_eq!(&base.as_slice()[3..], &[[0, 0]]);
        assert!(PaddedDataset::<u16>::new(1, layout).is_err());
    }

    #[test]
    fn cosine_view_preserves_byte_base_and_normalizes_only_logical_rows() {
        let base = PaddedDataset::from_packed(
            &[3u8, 4, 0, 0, 0, 5],
            PaddedLayout::new::<u8>(3, 32).unwrap(),
        )
        .unwrap();
        let normalized = TransformedRows {
            base: &base,
            transform: RowTransform::UnitL2,
        };
        let mut scratch = Vec::new();
        assert_eq!(normalized.row_f32(0, &mut scratch), &[0.6, 0.8, 0.]);
        assert_eq!(normalized.row_f32(1, &mut scratch), &[0., 0., 1.]);
        assert_eq!(base.row(0).unwrap(), &[3, 4, 0]);
    }
}
