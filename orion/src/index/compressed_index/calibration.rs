//! Orion calibration reuses the typed graph/statistics walk.
use super::{CompressedIndex, DynamicElement};
use crate::algorithm::search::calibrate::{
    CalibratedParams, CalibrationConfig, CalibrationDiagnostics, CalibrationMetric,
};
use diskann::{
    common::{ANNError, ANNResult},
    model::PaddedDataset,
};
use std::cell::RefCell;

pub(super) fn exact_distance(
    metric: CalibrationMetric,
    query: &[f32],
    row: &[f32],
) -> ANNResult<f32> {
    let distance = match metric {
        CalibrationMetric::L2 => vector::distance_l2_slice_f32(query, row),
        CalibrationMetric::InnerProduct => {
            // Preserve the compatibility facade's scalar accumulation order
            // for dimensions that previously used its odd-dimension fallback.
            if query.len() % 4 == 0 {
                vector::distance_ip_slice_f32(query, row)
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
fn error(message: &str) -> ANNError {
    ANNError::log_index_error(message.into())
}

impl<T: DynamicElement> CompressedIndex<PaddedDataset<T>> {
    fn calibration_inputs(&self, queries: &[f32], metric: CalibrationMetric) -> ANNResult<usize> {
        self.validate_dataset_graph()?;
        let dim = self.dataset.layout().logical_dim();
        if queries.len() % dim != 0 || queries.iter().any(|v| !v.is_finite()) {
            return Err(error(
                "invalid calibration query shape or non-finite coordinate",
            ));
        }
        if !T::supports_calibration(metric) {
            return Err(error("native u8 calibration requires L2"));
        }
        Ok(dim)
    }

    /// Packed logical f32 queries. Native-u8 calibration widens only the current
    /// visited row into reusable scratch; it never materializes an f32 base.
    pub fn calibrate(
        &self,
        queries: &[f32],
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> ANNResult<CalibratedParams> {
        let dim = self.calibration_inputs(queries, config.metric)?;
        let scratch = RefCell::new(vec![0.; dim]);
        self.calibrate_scored(queries.len() / dim, beam, window, config, |q, id| {
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

    pub fn calibrate_with_diagnostics(
        &self,
        queries: &[f32],
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> ANNResult<CalibrationDiagnostics> {
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
