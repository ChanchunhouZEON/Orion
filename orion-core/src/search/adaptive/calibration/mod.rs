/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Orion calibration reuses the typed graph/statistics walk.
use crate::index::CompressedIndex;
#[cfg(feature = "benchmark")]
use crate::search::adaptive::calibrate::CalibrationDiagnostics;
use crate::search::adaptive::calibrate::{CalibratedParams, CalibrationConfig};
use crate::{Error, Result};
use orion_ann_diskann::model::PaddedDataset;
use std::cell::RefCell;

mod query;
mod scoring;

use query::CalibrationQuery;
pub use scoring::CalibrationScalar;
use scoring::wide_distance;

#[cfg(test)]
mod tests;

/// Creates a consistent invalid-input error for calibration validation and scoring.
fn error(message: &str) -> Error {
    Error::invalid_argument(message)
}

impl<T: CalibrationScalar> CompressedIndex<PaddedDataset<T>> {
    /// Checks graph compatibility and the legacy f32 calibration input before walking the graph.
    #[cfg(feature = "benchmark")]
    fn calibration_inputs(&self, queries: &[f32]) -> Result<usize> {
        self.validate_dataset_graph()?;
        let dim = self.dataset.layout().logical_dim();
        if !queries.len().is_multiple_of(dim)
            || queries.iter().any(|&v| !T::compatible_coordinate(v))
        {
            return Err(error(
                "invalid calibration query shape or incompatible coordinate",
            ));
        }
        Ok(dim)
    }

    /// Calibrates from packed logical f32 queries through the shared typed-input path.
    ///
    /// For a byte base, exactly representable byte queries retain the native
    /// calibration arithmetic; other queries use mixed-coordinate scoring.
    /// Neither path materializes a second full-precision base.
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
        let prepared = query::prepare::<T>(queries, dim, config.metric)?;
        let scratch = RefCell::new(vec![0.; dim]);
        self.calibrate_scored(prepared.len(), beam, window, config, |q, id| {
            let row = self
                .dataset
                .row(id as usize)
                .ok_or_else(|| error("invalid graph vertex during calibration"))?;
            match &prepared[q] {
                CalibrationQuery::Compatible(q) => {
                    T::compatible_distance(config.metric, q, row, &mut scratch.borrow_mut())
                }
                CalibrationQuery::Wide(q) => wide_distance(config.metric, q, row),
            }
        })
    }

    /// Runs the dynamic benchmark calibration walk while retaining observations used to explain its parameters.
    #[cfg(feature = "benchmark")]
    pub fn calibrate_with_diagnostics(
        &self,
        queries: &[f32],
        beam: usize,
        window: usize,
        config: CalibrationConfig,
    ) -> Result<CalibrationDiagnostics> {
        let dim = self.calibration_inputs(queries)?;
        let scratch = RefCell::new(vec![0.; dim]);
        self.calibrate_diagnostics_scored(queries.len() / dim, beam, window, config, |q, id| {
            let row = self
                .dataset
                .row(id as usize)
                .ok_or_else(|| error("invalid graph vertex during calibration"))?;
            T::compatible_distance(
                config.metric,
                &queries[q * dim..(q + 1) * dim],
                row,
                &mut scratch.borrow_mut(),
            )
        })
    }
}
