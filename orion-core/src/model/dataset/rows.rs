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
    fn dimension(&self) -> usize;
    fn len(&self) -> usize;
    fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32];
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        let mut scratch = Vec::new();
        let row = self.row_f32(id, &mut scratch);
        output.clear();
        output.extend_from_slice(row);
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    fn dimension(&self) -> usize {
        self.layout().logical_dim()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn row_f32<'a>(&'a self, id: usize, _: &'a mut Vec<f32>) -> &'a [f32] {
        self.row(id).expect("row ID out of bounds")
    }
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
    fn dimension(&self) -> usize {
        N
    }
    fn len(&self) -> usize {
        self.num_active_pts
    }
    fn row_f32<'a>(&'a self, id: usize, _: &'a mut Vec<f32>) -> &'a [f32] {
        &self.data[id * N..(id + 1) * N]
    }
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
pub(crate) struct TransformedRows<'a, D> {
    pub base: &'a D,
    pub transform: RowTransform,
}
impl<D: Rows> Rows for TransformedRows<'_, D> {
    fn dimension(&self) -> usize {
        self.base.dimension()
    }
    fn len(&self) -> usize {
        self.base.len()
    }
    fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
        if self.transform == RowTransform::Identity {
            return self.base.row_f32(id, scratch);
        }
        self.copy_row_f32(id, scratch);
        scratch
    }
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
    if memory_stride == disk_stride {
        reader.read_exact(bytemuck::cast_slice_mut(data.as_mut_slice()))?;
        return Ok(data);
    }
    for row in data.as_mut_slice().chunks_exact_mut(memory_stride) {
        reader.read_exact(bytemuck::cast_slice_mut(&mut row[..disk_stride]))?;
    }
    Ok(data)
}

/// Full-precision stage factories for the two storage facades. Application
/// code is generic over storage; all new frontends instantiate padded rows.
pub trait CascadeData: Rows + crate::index::IndexData + Send {
    fn padded_f32(&self) -> Option<&PaddedDataset<f32>> {
        None
    }
    fn padded_u8(&self) -> Option<&PaddedDataset<u8>> {
        None
    }
    fn allocation_bytes(&self) -> &[u8];
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        epsilon: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>>;
    fn truth(
        &self,
        inner_product: bool,
    ) -> Box<dyn crate::search::cascade::stage::RerankStage + '_>;
}
impl CascadeData for PaddedDataset<f32> {
    fn padded_f32(&self) -> Option<&PaddedDataset<f32>> {
        Some(self)
    }
    fn allocation_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
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
    fn allocation_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
    }
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        epsilon: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>> {
        Ok(Box::new(
            crate::search::cascade::stage::admission::AdsF32Admission::new(self, epsilon),
        ))
    }
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
    fn padded_u8(&self) -> Option<&PaddedDataset<u8>> {
        Some(self)
    }
    fn allocation_bytes(&self) -> &[u8] {
        self.as_slice()
    }
    #[cfg(feature = "benchmark")]
    fn ads(
        &self,
        _: f32,
    ) -> crate::Result<Box<dyn crate::search::cascade::stage::AdmissionStage + '_>> {
        Err(crate::Error::unsupported(
            "ADSampling requires f32 base storage",
        ))
    }
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
