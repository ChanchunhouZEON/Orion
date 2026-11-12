/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Orion calibration reuses the typed graph/statistics walk.
use crate::index::{CompressedIndex, DynamicElement};
#[cfg(feature = "benchmark")]
use crate::search::adaptive::calibrate::CalibrationDiagnostics;
use crate::search::adaptive::calibrate::{CalibratedParams, CalibrationConfig, CalibrationMetric};
use crate::{Error, Result};
use orion_ann_diskann::model::PaddedDataset;
use std::cell::RefCell;

pub(crate) fn exact_distance(metric: CalibrationMetric, query: &[f32], row: &[f32]) -> Result<f32> {
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
fn error(message: &str) -> Error {
    Error::invalid_argument(message)
}

impl<T: DynamicElement> CompressedIndex<PaddedDataset<T>> {
    #[cfg(feature = "benchmark")]
    fn calibration_inputs(&self, queries: &[f32], metric: CalibrationMetric) -> Result<usize> {
        self.validate_dataset_graph()?;
        let dim = self.dataset.layout().logical_dim();
        if !queries.len().is_multiple_of(dim) || queries.iter().any(|&v| !T::valid_query(v)) {
            return Err(error(
                "invalid calibration query shape or non-finite coordinate",
            ));
        }
        if !T::supports_calibration(metric) {
            return Err(error("native u8 calibration requires L2"));
        }
        Ok(dim)
    }

    /// Packed logical f32 queries. Native-u8 calibration requires exact byte
    /// values; L2 widens one row into reusable scratch, while IP/cosine use wide
    /// integer accumulation. No metric materializes a second f32 base.
    pub fn calibrate(
        &self,
        queries: &[f32],
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> Result<CalibratedParams> {
        self.calibrate_queries(
            crate::search::query::QueryInput::F32(queries),
            beam,
            window,
            config,
        )
    }

    /// Typed calibration uses the same original-coordinate validation and
    /// per-row preparation as search. Homogeneous inputs retain their previous
    /// accumulation order; mixed/wide coordinates never narrow to base storage.
    pub fn calibrate_queries(
        &self,
        queries: crate::search::query::QueryInput<'_>,
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> Result<CalibratedParams> {
        self.validate_dataset_graph()?;
        let dim = self.dataset.layout().logical_dim();
        queries.validate()?;
        if queries.is_empty() || !queries.len().is_multiple_of(dim) {
            return Err(error("invalid calibration query shape"));
        }
        let prepared = (0..queries.len() / dim)
            .map(|i| {
                let q = queries.slice(i * dim..(i + 1) * dim);
                if config.metric == CalibrationMetric::Cosine && q.is_zero() {
                    return Err(error("cosine requires nonzero query vectors"));
                }
                if q.lossless_f32() {
                    let mut values = q.to_f32();
                    if q.needs_unit_norm() {
                        crate::search::query::normalize(values.to_mut())?;
                    }
                    if values.iter().all(|&x| T::valid_query(x)) {
                        return Ok(CalibrationQuery::Compatible(values));
                    }
                }
                let values = q.prepared_wide()?;
                Ok(CalibrationQuery::Wide(values))
            })
            .collect::<Result<Vec<_>>>()?;
        let scratch = RefCell::new(vec![0.; dim]);
        self.calibrate_scored(prepared.len(), beam, window, config, |q, id| {
            let row = self
                .dataset
                .row(id as usize)
                .ok_or_else(|| error("invalid graph vertex during calibration"))?;
            let distance = match &prepared[q] {
                CalibrationQuery::Compatible(q) => {
                    return T::calibration_distance(
                        config.metric,
                        q,
                        row,
                        &mut scratch.borrow_mut(),
                    );
                }
                CalibrationQuery::Wide(q) => {
                    let metric = match config.metric {
                        CalibrationMetric::SquaredL2 => crate::ScoreMetric::SquaredL2,
                        CalibrationMetric::InnerProduct => crate::ScoreMetric::InnerProduct,
                        CalibrationMetric::Cosine => crate::ScoreMetric::CosineSimilarity,
                    };
                    let value = crate::search::query::numeric::score(q, row, metric);
                    match config.metric {
                        CalibrationMetric::SquaredL2 => value,
                        CalibrationMetric::InnerProduct => -value,
                        CalibrationMetric::Cosine => 1.0 - value,
                    }
                }
            };
            if distance.is_finite() {
                Ok(distance)
            } else {
                Err(error("calibration distance must be finite"))
            }
        })
    }

    #[cfg(feature = "benchmark")]
    pub fn calibrate_with_diagnostics(
        &self,
        queries: &[f32],
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> Result<CalibrationDiagnostics> {
        let dim = self.calibration_inputs(queries, config.metric)?;
        let scratch = RefCell::new(vec![0.; dim]);
        self.calibrate_diagnostics_scored(queries.len() / dim, beam, window, config, |q, id| {
            let row = self
                .dataset
                .row(id as usize)
                .ok_or_else(|| error("invalid graph vertex during calibration"))?;
            T::calibration_distance(
                config.metric,
                &queries[q * dim..(q + 1) * dim],
                row,
                &mut scratch.borrow_mut(),
            )
        })
    }
}

enum CalibrationQuery<'a> {
    Compatible(std::borrow::Cow<'a, [f32]>),
    Wide(Vec<f64>),
}
