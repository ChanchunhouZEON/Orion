/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Original-metric calibration scores; quantized admission scores never enter here.
//!
//! Compatibility arithmetic deliberately preserves the historical f32/byte
//! accumulation order. Mixed queries use wide arithmetic instead of narrowing
//! their coordinates to the base representation.

use super::error;
use crate::model::dataset::base::BaseScalar;
use crate::{CalibrationMetric, Result};

/// Arithmetic capability used by core calibration and its runtime adapter.
///
/// This integration trait is sealed through BaseScalar. Storage metadata stays
/// on BaseScalar; only this module decides query compatibility and scoring.
/// Implementations receive validated, equal-length logical rows and reusable
/// scratch. They must return smaller-is-closer original-metric distances.
pub trait CalibrationScalar: BaseScalar {
    /// Chooses compatibility arithmetic without rejecting mixed-type queries.
    fn compatible_coordinate(value: f32) -> bool;

    /// Scores a query accepted by compatible_coordinate against one logical row.
    fn compatible_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> Result<f32>;
}

// All implementations come from this policy table. The macro only forwards
// calls; compatibility tests and numerical algorithms stay ordinary functions.
macro_rules! define_calibration_scalars {
    ($($scalar:ty => ($compatible:expr, $score:expr)),+ $(,)?) => {$(
        impl CalibrationScalar for $scalar {
            /// Selects the table's compatibility arithmetic without rejecting mixed inputs.
            fn compatible_coordinate(value: f32) -> bool {
                ($compatible)(value)
            }

            /// Scores one logical row using the scalar's declared numerical policy.
            fn compatible_distance(
                metric: CalibrationMetric,
                query: &[f32],
                row: &[Self],
                scratch: &mut [f32],
            ) -> Result<f32> {
                ($score)(metric, query, row, scratch)
            }
        }
    )+};
}

// Existing f32/u8 arithmetic is intentionally retained. Other integers widen
// directly to f64, preserving i32/u32 coordinates beyond f32's exact range.
define_calibration_scalars! {
    f32 => (f32::is_finite, |metric, query, row, _scratch| exact_distance(metric, query, row)),
    u8 => (byte_compatible_coordinate, byte_distance),
    i8 => (f32::is_finite, wide_compatible_distance),
    i16 => (f32::is_finite, wide_compatible_distance),
    i32 => (f32::is_finite, wide_compatible_distance),
    u16 => (f32::is_finite, wide_compatible_distance),
    u32 => (f32::is_finite, wide_compatible_distance),
}

/// Selects integer byte arithmetic only when every query coordinate can be exact.
fn byte_compatible_coordinate(value: f32) -> bool {
    value.is_finite() && (0.0..=255.0).contains(&value) && value.fract() == 0.0
}

/// Adapts wide arithmetic to the shared scoring signature without touching scratch.
fn wide_compatible_distance<T: BaseScalar>(
    metric: CalibrationMetric,
    query: &[f32],
    row: &[T],
    _: &mut [f32],
) -> Result<f32> {
    wide_distance(metric, query, row)
}

/// Converts a business score into calibration's ascending distance convention.
///
/// Both query and base coordinates widen directly to f64. Invalid zero cosine
/// norms or a non-finite f32 result are rejected before entering the graph walk.
pub(super) fn wide_distance<Q: Copy + Into<f64>, T: BaseScalar>(
    metric: CalibrationMetric,
    query: &[Q],
    row: &[T],
) -> Result<f32> {
    let score_metric = match metric {
        CalibrationMetric::SquaredL2 => crate::ScoreMetric::SquaredL2,
        CalibrationMetric::InnerProduct => crate::ScoreMetric::InnerProduct,
        CalibrationMetric::Cosine => crate::ScoreMetric::CosineSimilarity,
    };
    let value = crate::search::query::numeric::score(query, row, score_metric);
    let distance = match metric {
        CalibrationMetric::SquaredL2 => value,
        CalibrationMetric::InnerProduct => -value,
        CalibrationMetric::Cosine => 1.0 - value,
    };
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(error("calibration distance must be finite"))
    }
}

/// Preserves integer IP/cosine accumulation and the widened-f32 L2 reference path.
fn byte_distance(
    metric: CalibrationMetric,
    query: &[f32],
    row: &[u8],
    scratch: &mut [f32],
) -> Result<f32> {
    if metric != CalibrationMetric::SquaredL2 {
        let mut dot = 0u64;
        let (mut qnorm, mut norm) = (0u64, 0u64);
        for (&q, &x) in query.iter().zip(row) {
            let (q, x) = (q as u64, u64::from(x));
            dot += q * x;
            if metric == CalibrationMetric::Cosine {
                qnorm += q * q;
                norm += x * x;
            }
        }
        return if metric == CalibrationMetric::InnerProduct {
            Ok(-(dot as f32))
        } else {
            if qnorm == 0 || norm == 0 {
                return Err(error("cosine requires nonzero vectors"));
            }
            Ok(
                (1.0 - dot as f64 / ((qnorm as f64).sqrt() * (norm as f64).sqrt())).clamp(0., 1.)
                    as f32,
            )
        };
    }
    for (out, &v) in scratch.iter_mut().zip(row) {
        *out = f32::from(v);
    }
    exact_distance(metric, query, scratch)
}

/// Computes a finite ascending calibration distance using the original metric.
///
/// Cosine rejects zero-norm vectors; this helper does not use quantized admission scores.
fn exact_distance(metric: CalibrationMetric, query: &[f32], row: &[f32]) -> Result<f32> {
    let distance = match metric {
        CalibrationMetric::SquaredL2 => orion_ann_vector::distance_l2_slice_f32(query, row),
        CalibrationMetric::InnerProduct => {
            // Preserve the compatibility facade's scalar accumulation order
            // for dimensions that previously used its odd-dimension fallback.
            if query.len().is_multiple_of(4) {
                orion_ann_vector::distance_ip_slice_f32(query, row)
            } else {
                -query.iter().zip(row).map(|(x, y)| x * y).sum::<f32>()
            }
        }
        CalibrationMetric::Cosine => {
            let (mut dot, mut a, mut b) = (0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in query.iter().zip(row) {
                let (x, y) = (f64::from(x), f64::from(y));
                dot += x * y;
                a += x * x;
                b += y * y;
            }
            if a == 0. || b == 0. {
                return Err(error("cosine requires nonzero vectors"));
            }
            (1. - (dot / (a.sqrt() * b.sqrt())).clamp(-1., 1.)) as f32
        }
    };
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(error("calibration distance must be finite"))
    }
}
