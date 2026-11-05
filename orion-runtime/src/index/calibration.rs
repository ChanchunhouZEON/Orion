/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Per-k calibration for a resident index.
use super::{query::validate_search, Index};
use crate::Result;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Calibration for a specific index, cascade and top-k; included in snapshots.
pub struct Calibration {
    pub k: usize,
    pub search_list_size: usize,
    pub samples: usize,
    pub threshold: f32,
    pub early_exit_limit: usize,
}
impl Calibration {
    pub(super) fn params(&self) -> orion_ann_core::CalibratedParams {
        orion_ann_core::CalibratedParams {
            threshold: self.threshold,
            early_exit_limit: self.early_exit_limit,
        }
    }
}

impl Index {
    /// See the [complete example and contract](crate::guide::search).
    ///
    /// Calibrate a nonempty packed row-major query batch for a particular k.
    /// Prepares the selected sidecars under exclusive access before publishing
    /// calibration, so subsequent `&self` searches may run concurrently.
    ///
    /// # Errors
    /// Invalid shapes, values, controls and backend failures are returned.
    pub fn calibrate(
        &mut self,
        mut queries: Vec<f32>,
        k: usize,
        search_list_size: Option<usize>,
    ) -> Result<Calibration> {
        let beam = search_list_size
            .unwrap_or_else(|| orion_ann_core::calibration_search_list_size!(48, k));
        validate_search(self.len(), k, beam)?;
        if queries.is_empty() {
            return Err("calibration requires at least one query".into());
        }
        self.queries(&mut queries)?;
        let params = self.pool.install(|| {
            // Sidecar builders may use Rayon while holding their initialization
            // lock. Finish them under exclusive Index access, before allowing
            // concurrent searches to enter this worker pool.
            self.view.engine.prepare_search(&self.config)?;
            self.view.engine.calibrate(&queries, &self.config, k, beam)
        })?;
        let calibration = Calibration {
            k,
            search_list_size: beam,
            samples: queries.len() / self.dimension(),
            threshold: params.threshold,
            early_exit_limit: params.early_exit_limit,
        };
        self.calibrations.insert(k, calibration.clone());
        Ok(calibration)
    }

    /// Calibration currently available for adaptive searches of this k.
    #[cfg(feature = "cli")]
    pub fn calibration(&self, k: usize) -> Option<&Calibration> {
        self.calibrations.get(&k)
    }
}
