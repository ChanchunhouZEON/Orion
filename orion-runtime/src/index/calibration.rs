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
    /// Prepares sidecars and applies `auto_pin` under exclusive access before publishing
    /// calibration, so subsequent `&self` searches may run concurrently.
    ///
    /// # Errors
    /// Invalid shapes, values, controls and backend failures are returned.
    pub fn calibrate(
        &mut self,
        queries: Vec<f32>,
        k: usize,
        search_list_size: Option<usize>,
    ) -> Result<Calibration> {
        self.calibrate_batch(
            super::QueryBatch::from_f32(&queries, self.dimension())?,
            k,
            search_list_size,
        )
    }

    /// Calibrate borrowed queries of any supported scalar type. No batch-wide
    /// conversion is performed, and the borrow ends when calibration returns.
    pub fn calibrate_batch(
        &mut self,
        queries: super::QueryBatch<'_>,
        k: usize,
        search_list_size: Option<usize>,
    ) -> Result<Calibration> {
        self.calibrate_many_batches(queries, &[k], search_list_size)?
            .pop()
            .ok_or_else(|| crate::Error::internal("missing prepared calibration"))
    }

    /// Explicitly prepare several k values using one validated sample batch.
    /// Duplicate/empty k lists are rejected. Publish all calibration records
    /// together after success; failures leave previous records intact. Preparing
    /// sidecars/residency can still materialize storage before a later failure.
    pub fn calibrate_many(
        &mut self,
        queries: Vec<f32>,
        ks: &[usize],
        search_list_size: Option<usize>,
    ) -> Result<Vec<Calibration>> {
        self.calibrate_many_batches(
            super::QueryBatch::from_f32(&queries, self.dimension())?,
            ks,
            search_list_size,
        )
    }

    /// Prepare multiple k values from one borrowed batch, publishing together.
    pub fn calibrate_many_batches(
        &mut self,
        queries: super::QueryBatch<'_>,
        ks: &[usize],
        search_list_size: Option<usize>,
    ) -> Result<Vec<Calibration>> {
        if queries.is_empty() || ks.is_empty() {
            return Err(crate::Error::invalid_argument(
                "calibration requires queries and at least one k",
            ));
        }
        let mut unique = std::collections::BTreeSet::new();
        let controls = ks
            .iter()
            .map(|&k| {
                if !unique.insert(k) {
                    return Err(crate::Error::invalid_argument("duplicate calibration k")
                        .with_detail("k", k));
                }
                let beam = search_list_size
                    .unwrap_or_else(|| orion_ann_core::calibration_search_list_size!(48, k));
                validate_search(self.len(), k, beam)?;
                Ok((k, beam))
            })
            .collect::<Result<Vec<_>>>()?;
        let queries = self.prepare_queries(queries)?;
        let samples = queries.len();
        let _lease = self
            .working_directory
            .as_deref()
            .map(crate::storage::WriterLease::acquire)
            .transpose()?;
        let prepared = self.pool.install(|| {
            self.view.engine.prepare_search(&self.config)?;
            controls
                .into_iter()
                .map(|(k, beam)| {
                    let params =
                        self.view
                            .engine
                            .calibrate(queries.values, &self.config, k, beam)?;
                    Ok(Calibration {
                        k,
                        search_list_size: beam,
                        samples,
                        threshold: params.threshold,
                        early_exit_limit: params.early_exit_limit,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for calibration in &prepared {
            self.calibrations.insert(calibration.k, calibration.clone());
        }
        Ok(prepared)
    }

    /// Prepared k values, in increasing order. Does not perform any work.
    pub fn calibrated_ks(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.calibrations.keys().copied()
    }

    /// Calibration currently available for adaptive searches of this k.
    pub fn calibration(&self, k: usize) -> Option<&Calibration> {
        self.calibrations.get(&k)
    }
}
