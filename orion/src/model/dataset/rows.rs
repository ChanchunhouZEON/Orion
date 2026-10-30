//! Borrow logical rows from either storage layout, without packing a second base.
use diskann::model::{InmemDataset, PaddedDataset};

pub trait F32Rows: Sync {
    fn dimension(&self) -> usize;
    fn len(&self) -> usize;
    fn row(&self, id: usize) -> &[f32];
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn range(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for id in 0..self.len() {
            for &value in self.row(id) {
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
impl<const N: usize> F32Rows for InmemDataset<f32, N> {
    fn dimension(&self) -> usize {
        N
    }
    fn len(&self) -> usize {
        self.num_active_pts
    }
    fn row(&self, id: usize) -> &[f32] {
        &self.data[id * N..(id + 1) * N]
    }
}
impl F32Rows for PaddedDataset<f32> {
    fn dimension(&self) -> usize {
        self.layout().logical_dim()
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn row(&self, id: usize) -> &[f32] {
        self.row(id).expect("row ID out of bounds")
    }
}

/// In-memory code rows contain complete distance-kernel blocks. Disk rows keep
/// the historical format; readers reconstruct zero padding for this machine.
pub fn write_rows<T: bytemuck::Pod>(
    writer: &mut impl std::io::Write,
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
pub fn read_rows<T: bytemuck::Pod>(
    reader: &mut impl std::io::Read,
    rows: usize,
    memory_stride: usize,
    disk_stride: usize,
) -> diskann::common::ANNResult<diskann::common::AlignedBoxWithSlice<T>> {
    use std::io::{Error, ErrorKind};
    let count = rows
        .checked_mul(memory_stride)
        .ok_or_else(|| Error::new(ErrorKind::InvalidData, "sidecar allocation overflow"))?;
    if rows == 0 || disk_stride == 0 || disk_stride > memory_stride {
        return Err(Error::new(ErrorKind::InvalidData, "invalid sidecar row layout").into());
    }
    let mut data = diskann::common::AlignedBoxWithSlice::<T>::new(count, 64)?;
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
pub trait CascadeData: F32Rows + crate::index::compressed_index::IndexData + Send {
    fn allocation(&self) -> &[f32];
    fn ads(&self, epsilon: f32) -> Box<dyn crate::algorithm::search::stage::AdmissionStage + '_>;
    fn truth(
        &self,
        inner_product: bool,
    ) -> Box<dyn crate::algorithm::search::stage::RerankStage + '_>;
}
impl CascadeData for PaddedDataset<f32> {
    fn allocation(&self) -> &[f32] {
        self.as_slice()
    }
    fn ads(&self, epsilon: f32) -> Box<dyn crate::algorithm::search::stage::AdmissionStage + '_> {
        Box::new(
            crate::algorithm::search::stage::admission::ads_f32::DynamicAdsF32Admission::new(
                self, epsilon,
            ),
        )
    }
    fn truth(&self, ip: bool) -> Box<dyn crate::algorithm::search::stage::RerankStage + '_> {
        use crate::algorithm::search::stage::rerank::DynamicF32Rerank;
        if ip {
            Box::new(DynamicF32Rerank::<vector::IpF32Distance>::new(self))
        } else {
            Box::new(DynamicF32Rerank::<vector::L2F32Distance>::new(self))
        }
    }
}
impl<const N: usize> CascadeData for InmemDataset<f32, N> {
    fn allocation(&self) -> &[f32] {
        self.data.as_slice()
    }
    fn ads(&self, epsilon: f32) -> Box<dyn crate::algorithm::search::stage::AdmissionStage + '_> {
        Box::new(crate::algorithm::search::stage::admission::AdsF32Admission::new(self, epsilon))
    }
    fn truth(&self, ip: bool) -> Box<dyn crate::algorithm::search::stage::RerankStage + '_> {
        use crate::algorithm::search::stage::rerank::{F32Rerank, IpF32Rerank};
        if ip {
            Box::new(IpF32Rerank::new(self))
        } else {
            Box::new(F32Rerank::new(self))
        }
    }
}
