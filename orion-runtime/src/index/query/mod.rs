/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Query validation, calibration and dispatch on the pinned resident generation.
mod input;
pub use input::QueryBatch;

use super::Index;
use super::SearchResult;
use crate::{Error, Result, VectorStorageKind};
use orion_ann_diagnostics::{RecordingObserver, SearchTrace, TraceLimits, TraceMetadata};

/// Per-request search controls. Calibration for `k` must already exist.
/// These options never alter the index's build recipe or saved configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchOptions {
    /// Maximum number of neighbors per query.
    pub k: usize,
    /// Candidate queue capacity; must be at least `k`.
    pub search_list_size: usize,
    /// Compute original-metric scores for the returned IDs. Default: IDs only.
    pub scores: bool,
}

impl SearchOptions {
    /// Construct controls; index-dependent bounds are checked by `Index::search`.
    pub const fn new(k: usize, search_list_size: usize) -> Self {
        Self {
            k,
            search_list_size,
            scores: false,
        }
    }

    /// Include original-metric scores without changing selected IDs or ordering.
    /// Reuse exact rerank scores when possible; otherwise score only returned IDs.
    pub const fn with_scores(mut self, enabled: bool) -> Self {
        self.scores = enabled;
        self
    }
}

impl Index {
    /// See the [complete example and contract](crate::guide::search).
    ///
    /// Search packed row-major queries, preserving input order in the results.
    /// Calibration for this `k` is required, including for empty batches.
    ///
    /// # Errors
    /// Returns `InvalidState` if calibration is missing, `DimensionMismatch`
    /// for a malformed packed batch, or an input/backend error. Rejected input
    /// does not invalidate the index; callers may correct it and retry.
    pub fn search(&self, queries: QueryBatch<'_>, options: &SearchOptions) -> Result<SearchResult> {
        self.search_batch(queries, options, true)
    }

    /// Trace one logical query using the same normalization, calibration and graph
    /// loop as `search`. Recording truncation never stops the underlying search.
    /// This opt-in path allocates diagnostic payloads and must not be timed as QPS.
    ///
    /// # Errors
    /// Returns the ordinary search validation/calibration errors, or a trace-limit
    /// error before search if the recording budget cannot hold k result IDs.
    pub fn trace_search(
        &mut self,
        query: Vec<f32>,
        k: usize,
        beam: usize,
        adaptive: bool,
        limits: TraceLimits,
    ) -> Result<SearchTrace> {
        self.trace_query(
            QueryBatch::from_f32(&query, self.dimension())?,
            k,
            beam,
            adaptive,
            limits,
        )
    }

    /// Trace one borrowed query of any supported type through ordinary preparation.
    pub fn trace_query(
        &mut self,
        query: QueryBatch<'_>,
        k: usize,
        beam: usize,
        adaptive: bool,
        limits: TraceLimits,
    ) -> Result<SearchTrace> {
        validate_search(self.len(), k, beam)?;
        if query.len() != 1 {
            return Err(Error::dimension("query rows", 1, query.len()).at("trace"));
        }
        let query = self.prepare_queries(query)?;
        let params = self.search_params(k, adaptive)?;
        let stages = self.config.cascade();
        let mut observer = RecordingObserver::new(
            TraceMetadata {
                dimension: self
                    .dimension()
                    .try_into()
                    .map_err(|_| Error::invalid_argument("dimension exceeds trace schema"))?,
                points: self.len() as u32,
                k: k as u32,
                beam: beam
                    .try_into()
                    .map_err(|_| Error::invalid_argument("beam exceeds trace schema"))?,
                adaptive,
                metric: format!("{:?}", self.config.metric),
                storage: format!("{:?}", self.storage()),
                prefilter: format!("{:?}", stages.prefilter),
                admission: format!("{:?}", stages.admission),
                rerank: format!("{:?}", stages.rerank),
            },
            limits,
        )?;
        let ids = self.pool.install(|| {
            self.view.engine.search_observed(
                query.values,
                &self.config,
                k,
                beam,
                params,
                &mut observer,
            )
        })?;
        observer.finish(ids)
    }

    pub(crate) fn search_batch(
        &self,
        queries: QueryBatch<'_>,
        options: &SearchOptions,
        adaptive: bool,
    ) -> Result<SearchResult> {
        validate_search(self.len(), options.k, options.search_list_size)?;
        let queries = self.prepare_queries(queries)?;
        let params = self.search_params(options.k, adaptive)?;
        let mut result: SearchResult = self
            .pool
            .install(|| {
                self.view
                    .engine
                    .search(queries, &self.config, options, params)
            })?
            .into();
        self.attach_external_ids(&mut result);
        Ok(result)
    }

    pub(super) fn attach_external_ids(&self, result: &mut SearchResult) {
        if let Some(mapping) = &self.external_ids {
            let mut ids = vec![u64::MAX; result.ids.len()];
            for (row, &count) in result.counts.iter().enumerate() {
                for column in 0..count as usize {
                    let slot = row * result.k + column;
                    ids[slot] = mapping.0[result.ids[slot] as usize];
                }
            }
            result.external_ids = Some(ids);
        }
    }

    pub(super) fn prepare_queries<'a>(&self, mut batch: QueryBatch<'a>) -> Result<QueryBatch<'a>> {
        if batch.dimension != self.dimension() {
            return Err(
                Error::dimension("queries.dimension", self.dimension(), batch.dimension)
                    .at("read_queries"),
            );
        }
        batch.values.validate()?;
        if self.config.metric == crate::SearchMetric::Cosine {
            if (0..batch.len()).any(|row| {
                batch
                    .values
                    .slice(row * batch.dimension..(row + 1) * batch.dimension)
                    .is_zero()
            }) {
                return Err(Error::invalid_argument(
                    "cosine requires nonzero query vectors",
                ));
            }
            if self.storage() == VectorStorageKind::F32 {
                batch.values = batch.values.with_unit_norm();
            }
        }
        Ok(batch)
    }

    fn search_params(&self, k: usize, adaptive: bool) -> Result<orion_ann_core::CalibratedParams> {
        Ok(if adaptive {
            self.calibrations
                .get(&k)
                .ok_or_else(|| {
                    Error::invalid_state(format!(
                        "calibrate this index for k={k} before adaptive search"
                    ))
                    .at("search")
                    .with_detail("k", k)
                    .with_detail("available_k", self.calibrated_ks().collect::<Vec<_>>())
                })?
                .params()
        } else {
            // Admission rates are nonnegative, so a zero threshold disables
            // convergence; early exit cannot fire before convergence either.
            orion_ann_core::CalibratedParams {
                threshold: 0.0,
                early_exit_limit: usize::MAX,
            }
        })
    }
}

pub(super) fn validate_search(points: usize, k: usize, beam: usize) -> Result<()> {
    if k == 0 || k > points || beam < k || beam > u32::MAX as usize / 4 {
        return Err(Error::invalid_argument(
            "require 1 <= k <= num_points and k <= search_list_size <= u32::MAX/4",
        )
        .with_detail("k", k)
        .with_detail("num_points", points)
        .with_detail("search_list_size", beam));
    }
    Ok(())
}
