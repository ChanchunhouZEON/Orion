/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validated query representations shared by calibration's graph walk.

use super::{error, scoring::CalibrationScalar};
use crate::search::query::QueryInput;
use crate::{CalibrationMetric, Result};
use std::borrow::Cow;

/// One logical query, retaining compatibility arithmetic only when lossless.
pub(super) enum CalibrationQuery<'a> {
    /// Borrowed or normalized f32 coordinates accepted by the base's arithmetic.
    Compatible(Cow<'a, [f32]>),
    /// Mixed or wide coordinates that must not be narrowed to native base values.
    Wide(Vec<f64>),
}

/// Validates a packed batch and chooses arithmetic independently for each row.
///
/// Normalization follows QueryInput's existing representation contract. Invalid
/// shapes, non-finite coordinates and zero cosine queries fail before traversal.
pub(super) fn prepare<T: CalibrationScalar>(
    queries: QueryInput<'_>,
    dim: usize,
    metric: CalibrationMetric,
) -> Result<Vec<CalibrationQuery<'_>>> {
    queries.validate()?;
    if queries.is_empty() || !queries.len().is_multiple_of(dim) {
        return Err(error("invalid calibration query shape"));
    }
    (0..queries.len() / dim)
        .map(|i| {
            let q = queries.slice(i * dim..(i + 1) * dim);
            if metric == CalibrationMetric::Cosine && q.is_zero() {
                return Err(error("cosine requires nonzero query vectors"));
            }
            if q.lossless_f32() {
                let mut values = q.to_f32();
                if q.needs_unit_norm() {
                    crate::search::query::normalize(values.to_mut())?;
                }
                if values.iter().all(|&x| T::compatible_coordinate(x)) {
                    return Ok(CalibrationQuery::Compatible(values));
                }
            }
            let values = q.prepared_wide()?;
            Ok(CalibrationQuery::Wide(values))
        })
        .collect::<Result<Vec<_>>>()
}
