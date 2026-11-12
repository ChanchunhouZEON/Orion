/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Query input, validation and per-row compute preparation. Conversion is
//! independent of base storage; padding copies an already selected representation.
use crate::{Error, Result};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::PaddedLayout;

pub mod controls;
mod input;
pub use input::{QueryBuffer, QueryElement, QueryInput};
pub(crate) mod numeric;
pub(crate) mod prepared;

/// Copy directly into the final padded row. Scalar width is a layout property,
/// not an input validation or conversion policy.
pub(crate) fn pad<T: bytemuck::Pod>(query: &[T], layout: PaddedLayout) -> Result<PaddedDataset<T>> {
    if query.len() != layout.logical_dim() {
        return Err(Error::dimension(
            "queries.dimension",
            layout.logical_dim(),
            query.len(),
        ));
    }
    let mut padded = PaddedDataset::new(1, layout).map_err(Error::invalid_argument)?;
    padded
        .row_mut(0)
        .expect("one allocated query row")
        .copy_from_slice(query);
    Ok(padded)
}

/// Normalize a private compute copy; caller coordinates are never mutated.
pub(crate) fn normalize(values: &mut [f32]) -> Result<()> {
    let norm = values
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(Error::invalid_argument(
            "cosine requires finite nonzero queries",
        ));
    }
    for value in values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}
