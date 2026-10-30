//! Input preparation for runtime-dimensional distance evaluation.
//!
//! File and NumPy inputs share the same layout. File formats stay packed;
//! only the final resident allocation has row padding.

use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

use bytemuck::Pod;
use diskann::model::PaddedDataset;
use vector::PaddedLayout;

use super::{
    inspect_vector_file, read_native_bytes, read_vector_row, BaseElement, VectorFormat,
    VECTOR_READ_BUFFER_BYTES,
};
use crate::search::cascade::SearchMetric;

/// Array/NumPy preparation. Normalization visits only logical coordinates;
/// zeroed row tails never enter quantizer statistics or normalization.
pub fn from_packed<T: BaseElement + Pod>(
    values: &[T],
    layout: PaddedLayout,
    metric: SearchMetric,
) -> Result<PaddedDataset<T>, String> {
    let mut data = PaddedDataset::from_packed(values, layout)?;
    normalize(&mut data, metric)?;
    Ok(data)
}

/// Bounded-buffer loading directly into the final padded allocation. Budget
/// includes resident storage and the bounded I/O scratch (not other index
/// components). Callers can use `layout.allocation_bytes` in full preflight.
pub fn load<T: BaseElement + Pod>(
    path: &Path,
    format: VectorFormat,
    layout: PaddedLayout,
    max_points: usize,
    metric: SearchMetric,
    memory_budget_bytes: Option<usize>,
) -> Result<PaddedDataset<T>, String> {
    let header = inspect_vector_file(path, format)?;
    if header.dimension != layout.logical_dim() {
        return Err(format!(
            "file dimension {} != logical dimension {}",
            header.dimension,
            layout.logical_dim()
        ));
    }
    let rows = header.count.min(max_points);
    let resident = layout.allocation_bytes(rows)?;
    let row_bytes = layout
        .logical_dim()
        .checked_mul(format.coordinate_bytes())
        .ok_or("encoded row size overflow")?;
    let peak = resident
        .checked_add(VECTOR_READ_BUFFER_BYTES)
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or("load size overflow")?;
    if memory_budget_bytes.is_some_and(|budget| peak > budget) {
        return Err(format!(
            "padded load needs {resident} resident bytes ({} padding bytes), \
            {} I/O buffer bytes and {row_bytes} row scratch bytes; exceeds budget {}",
            layout.padding_bytes(rows)?,
            VECTOR_READ_BUFFER_BYTES,
            memory_budget_bytes.unwrap()
        ));
    }
    let mut data = PaddedDataset::new(rows, layout)?;
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if format.has_file_header() {
        file.seek(SeekFrom::Start(8)).map_err(|e| e.to_string())?;
    }
    // Preserve the direct large-read path for already aligned native u8.
    if format == VectorFormat::U8bin {
        if let Some(bytes) = data.packed_mut().and_then(T::native_bytes_mut) {
            read_native_bytes(&mut file, bytes).map_err(|e| e.to_string())?;
            normalize(&mut data, metric)?;
            return Ok(data);
        }
    }
    let mut reader = BufReader::with_capacity(VECTOR_READ_BUFFER_BYTES, file);
    let mut encoded = vec![0; row_bytes];
    for id in 0..rows {
        let row = data.row_mut(id).ok_or("missing destination row")?;
        read_vector_row(&mut reader, format, layout.logical_dim(), &mut encoded, row)?;
        T::normalize(row, layout.logical_dim(), metric)?;
    }
    Ok(data)
}

fn normalize<T: BaseElement + Pod>(
    data: &mut PaddedDataset<T>,
    metric: SearchMetric,
) -> Result<(), String> {
    let dim = data.layout().logical_dim();
    for id in 0..data.len() {
        T::normalize(data.row_mut(id).ok_or("missing row")?, dim, metric)?;
    }
    Ok(())
}
