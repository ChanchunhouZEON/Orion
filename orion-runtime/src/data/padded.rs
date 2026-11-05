/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Input preparation for runtime-dimensional distance evaluation.
//!
//! File and NumPy inputs share the same layout. File formats stay packed;
//! only the final resident allocation has row padding.

use crate::integrity::HashedFile;
use crate::{Error, Result};
use std::io::{BufReader, Read};
use std::path::Path;

use bytemuck::Pod;
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::PaddedLayout;

use super::{
    inspect_vector_file, read_native_bytes, read_vector_row, BaseElement, VectorFormat,
    VECTOR_READ_BUFFER_BYTES,
};
use crate::config::cascade::SearchMetric;

/// Array/NumPy preparation. Normalization visits only logical coordinates;
/// zeroed row tails never enter quantizer statistics or normalization.
#[cfg(test)]
pub fn from_packed<T: BaseElement + Pod>(
    values: &[T],
    layout: PaddedLayout,
    metric: SearchMetric,
) -> Result<PaddedDataset<T>> {
    let mut data = PaddedDataset::from_packed(values, layout)?;
    normalize(&mut data, metric)?;
    Ok(data)
}

/// Bounded-buffer loading directly into the final padded allocation. Budget
/// includes resident storage and the bounded I/O scratch (not other index
/// components). Callers can use `layout.allocation_bytes` in full preflight.
#[cfg(any(test, feature = "cli", feature = "benchmark"))]
pub fn load<T: BaseElement + Pod>(
    path: &Path,
    format: VectorFormat,
    layout: PaddedLayout,
    max_points: usize,
    metric: SearchMetric,
    memory_budget_bytes: Option<usize>,
) -> Result<PaddedDataset<T>> {
    let mut file = std::fs::File::open(path)?;
    load_inner(
        path,
        format,
        layout,
        max_points,
        metric,
        memory_budget_bytes,
        &mut file,
        false,
    )
}

pub(crate) fn load_hashed<T: BaseElement + Pod>(
    path: &Path,
    format: VectorFormat,
    layout: PaddedLayout,
    max_points: usize,
    metric: SearchMetric,
) -> Result<(PaddedDataset<T>, String)> {
    let mut source = HashedFile::new(std::fs::File::open(path)?);
    let dataset = load_inner(
        path,
        format,
        layout,
        max_points,
        metric,
        None,
        &mut source,
        true,
    )?;
    Ok((dataset, source.finish().1))
}

#[allow(clippy::too_many_arguments)]
fn load_inner<T: BaseElement + Pod>(
    path: &Path,
    format: VectorFormat,
    layout: PaddedLayout,
    max_points: usize,
    metric: SearchMetric,
    memory_budget_bytes: Option<usize>,
    mut source: impl Read,
    hash_entire_file: bool,
) -> Result<PaddedDataset<T>> {
    let header = inspect_vector_file(path, format)?;
    if header.dimension != layout.logical_dim() {
        return Err(Error::from(format!(
            "file dimension {} != logical dimension {}",
            header.dimension,
            layout.logical_dim()
        )));
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
        return Err(Error::from(format!(
            "padded load needs {resident} resident bytes ({} padding bytes), \
            {} I/O buffer bytes and {row_bytes} row scratch bytes; exceeds budget {}",
            layout.padding_bytes(rows)?,
            VECTOR_READ_BUFFER_BYTES,
            memory_budget_bytes.unwrap()
        )));
    }
    let mut data = PaddedDataset::new(rows, layout)?;

    if format.has_file_header() {
        let mut bytes = [0; 8];
        source.read_exact(&mut bytes)?;
        if bytes[..4] != (header.count as u32).to_le_bytes()
            || bytes[4..] != (header.dimension as u32).to_le_bytes()
        {
            return Err(Error::invalid_data("vector header changed during load"));
        }
    }
    if let Some(bytes) = (format == VectorFormat::U8bin)
        .then(|| data.packed_mut().and_then(T::native_bytes_mut))
        .flatten()
    {
        read_native_bytes(&mut source, bytes)?;
        normalize(&mut data, metric)?;
    } else {
        let mut reader = BufReader::with_capacity(VECTOR_READ_BUFFER_BYTES, &mut source);
        let mut encoded = vec![0; row_bytes];
        for id in 0..rows {
            let row = data.row_mut(id).ok_or("missing destination row")?;
            read_vector_row(&mut reader, format, layout.logical_dim(), &mut encoded, row)?;
            T::normalize(row, layout.logical_dim(), metric)?;
        }
    }
    // A prefix index still identifies the complete source, including unused rows.
    if hash_entire_file {
        if rows < header.count {
            std::io::copy(&mut source, &mut std::io::sink())?;
        } else if source.read(&mut [0])? != 0 {
            return Err(Error::invalid_data("base grew during load"));
        }
    }
    Ok(data)
}

fn normalize<T: BaseElement + Pod>(
    data: &mut PaddedDataset<T>,
    metric: SearchMetric,
) -> Result<()> {
    let dim = data.layout().logical_dim();
    for id in 0..data.len() {
        T::normalize(data.row_mut(id).ok_or("missing row")?, dim, metric)?;
    }
    Ok(())
}
