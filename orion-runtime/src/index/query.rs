/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Query validation, calibration and dispatch on the pinned resident generation.
mod input;
pub use input::QueryBatch;
pub(super) use input::QueryValues;

use super::Index;
use crate::{data::validate_and_normalize_vectors, Error, Result, VectorStorageKind};
use orion_ann_core::SearchResult;
use orion_ann_diagnostics::{RecordingObserver, SearchTrace, TraceLimits, TraceMetadata};

/// Per-request search controls. Calibration for `k` must already exist.
/// These options never alter the index's build recipe or saved configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchOptions {
    /// Maximum number of neighbors per query.
    pub k: usize,
    /// Candidate queue capacity; must be at least `k`.
    pub search_list_size: usize,
}

impl SearchOptions {
    /// Construct controls; index-dependent bounds are checked by `Index::search`.
    pub const fn new(k: usize, search_list_size: usize) -> Self {
        Self {
            k,
            search_list_size,
        }
    }
}

// An allocation is required only for representation conversion or normalization.
// The original caller's coordinates are never modified, including for cosine.
enum PreparedQueries<'a> {
    Borrowed(QueryBatch<'a>),
    Converted(Vec<f32>, usize),
}
impl PreparedQueries<'_> {
    fn batch(&self) -> QueryBatch<'_> {
        match self {
            Self::Borrowed(batch) => *batch,
            Self::Converted(values, dimension) => QueryBatch {
                values: QueryValues::F32(values),
                dimension: *dimension,
            },
        }
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
        mut query: Vec<f32>,
        k: usize,
        beam: usize,
        adaptive: bool,
        limits: TraceLimits,
    ) -> Result<SearchTrace> {
        validate_search(self.len(), k, beam)?;
        if query.len() != self.dimension() {
            return Err(Error::dimension("query", self.dimension(), query.len()).at("trace"));
        }
        self.queries(&mut query)?;
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
            self.view
                .engine
                .search_observed(&query, &self.config, k, beam, params, &mut observer)
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
        self.pool.install(|| {
            self.view
                .engine
                .search(queries.batch(), &self.config, options, params)
        })
    }

    fn prepare_queries<'a>(&self, batch: QueryBatch<'a>) -> Result<PreparedQueries<'a>> {
        if batch.dimension != self.dimension() {
            return Err(
                Error::dimension("queries.dimension", self.dimension(), batch.dimension)
                    .at("read_queries"),
            );
        }
        let mut values = match batch.values {
            QueryValues::U8(_) if self.storage() == VectorStorageKind::U8 => {
                return Ok(PreparedQueries::Borrowed(batch))
            }
            QueryValues::U8(values) => values.iter().map(|&v| f32::from(v)).collect(),
            QueryValues::F32(values) => {
                if self.config.metric != crate::SearchMetric::Cosine {
                    // L2 and IP require no normalization: validate without copying.
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(Error::invalid_argument("non-finite vector coordinate"));
                    }
                    if self.storage() == VectorStorageKind::U8
                        && values
                            .iter()
                            .any(|v| *v < 0. || *v > 255. || v.fract() != 0.)
                    {
                        return Err(Error::invalid_argument(
                            "native u8 queries must contain exact byte values",
                        ));
                    }
                    return Ok(PreparedQueries::Borrowed(batch));
                }
                values.to_vec()
            }
        };
        self.queries(&mut values)?;
        Ok(PreparedQueries::Converted(values, batch.dimension))
    }

    pub(super) fn queries(&self, queries: &mut [f32]) -> Result<()> {
        if queries.len() % self.dimension() != 0 {
            return Err(
                Error::packed_shape("queries", self.dimension(), queries.len()).at("read_queries"),
            );
        }
        if self.storage() == VectorStorageKind::U8
            && queries
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 255. || v.fract() != 0.)
        {
            return Err("native u8 queries must contain exact byte values".into());
        }
        validate_and_normalize_vectors(queries, self.dimension(), self.config.metric)
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
