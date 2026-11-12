/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Cold-path validation shared by sidecar constructors and cache readers.
use super::rows::Rows;
use crate::{Error, Result};

pub(super) fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| Error::invalid_data("sidecar size exceeds the addressable allocation limit"))
}

/// Check the whole payload before trusting a header to size an allocation.
/// Each pair is (count, bytes per item); no intermediate product can wrap.
pub(super) fn file_length(actual: u64, header: usize, parts: &[(usize, usize)]) -> Result<()> {
    let mut expected = header;
    for &(count, width) in parts {
        expected = expected
            .checked_add(product(count, width)?)
            .ok_or_else(|| Error::invalid_data("sidecar file length overflow"))?;
    }
    if expected as u64 != actual {
        return Err(
            Error::invalid_data("sidecar payload length does not match its header")
                .with_detail("expected_bytes", expected)
                .with_detail("actual_bytes", actual),
        );
    }
    Ok(())
}

pub(super) fn shape(rows: usize, dimension: usize, max_dimension: usize) -> Result<()> {
    if rows == 0
        || rows > u32::MAX as usize
        || dimension == 0
        || dimension > max_dimension
        || dimension > u32::MAX as usize
    {
        return Err(Error::invalid_argument("unsupported sidecar shape")
            .with_detail("rows", rows)
            .with_detail("dimension", dimension)
            .with_detail("max_dimension", max_dimension));
    }
    product(rows, dimension)?;
    Ok(())
}

pub(super) fn positive_scale(scale: f32) -> Result<()> {
    let square = scale * scale;
    if !scale.is_finite() || scale <= 0. || !square.is_finite() || square == 0. {
        return Err(Error::invalid_data(
            "quantization scale and its square must be finite and positive",
        ));
    }
    Ok(())
}

pub(super) fn finite(values: &[f32], nonnegative: bool) -> Result<()> {
    if values
        .iter()
        .any(|&v| !v.is_finite() || (nonnegative && v < 0.))
    {
        return Err(Error::invalid_data(
            "sidecar contains invalid floating-point parameters",
        ));
    }
    Ok(())
}

/// Borrow rows, validate their shape and coordinates, and cache the fitted range.
/// This does not trust an externally implemented `Rows::range` override.
/// It replaces the range scan already required by affine quantizers.
pub(super) struct CheckedRows<'a, D> {
    data: &'a D,
    dimension: usize,
    rows: usize,
    range: (f32, f32),
}
impl<'a, D: Rows> CheckedRows<'a, D> {
    pub(super) fn new(data: &'a D, max_dimension: usize) -> Result<Self> {
        let dimension = data.dimension();
        let rows = data.len();
        shape(rows, dimension, max_dimension)?;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut scratch = Vec::new();
        for id in 0..rows {
            let row = data.row_f32(id, &mut scratch);
            if row.len() != dimension {
                return Err(
                    Error::dimension("sidecar row", dimension, row.len()).with_detail("row", id)
                );
            }
            for &value in row {
                if !value.is_finite() {
                    return Err(Error::invalid_argument(
                        "quantization requires finite coordinates",
                    )
                    .with_detail("row", id));
                }
                min = min.min(value);
                max = max.max(value);
            }
        }
        Ok(Self {
            data,
            dimension,
            rows,
            range: (min, max),
        })
    }
}
impl<D: Rows> Rows for CheckedRows<'_, D> {
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn len(&self) -> usize {
        self.rows
    }
    fn copy_row_f32(&self, id: usize, output: &mut Vec<f32>) {
        self.data.copy_row_f32(id, output);
        assert_eq!(
            output.len(),
            self.dimension,
            "row provider changed shape during construction"
        );
    }
    fn range(&self) -> (f32, f32) {
        self.range
    }
    fn row_f32<'a>(&'a self, id: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
        let row = self.data.row_f32(id, scratch);
        // A custom row provider can change between calls. Never let a shorter
        // second row reach raw SIMD code, even after the initial validation.
        assert_eq!(
            row.len(),
            self.dimension,
            "row provider changed shape during construction"
        );
        row
    }
}
