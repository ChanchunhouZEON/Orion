/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! One executor for quantized and identity representations. Coordinate/metric
//! dispatch happens before this layer; neither scheduling nor beam traversal
//! inspects the base element type.
use crate::index::IndexData;
use crate::model::search::scratch::InMemSearchScratch;
use crate::search::{
    cascade::stage::{AdmissionSession, PrefilterSession},
    diagnostics::{NeighborMode, NoopObserver, SearchObserver, metrics::SetupTimer},
    query::controls::DynamicSearchConfig,
};
use crate::{CompressedIndex, Result, ScoreMetric, SearchResult};
use rayon::prelude::*;

pub(super) use crate::search::output::SearchRow;

/// Borrow the actual session, not a trait object around its Box. This avoids
/// introducing a second virtual call for existing quantized stage factories.
pub(crate) trait SessionHandle {
    /// Borrows the underlying admission session without adding a second trait-object wrapper.
    fn session(&self) -> &dyn AdmissionSession;
}
impl<T: AdmissionSession> SessionHandle for T {
    /// Borrows the underlying admission session without adding a second trait-object wrapper.
    fn session(&self) -> &dyn AdmissionSession {
        self
    }
}
impl SessionHandle for Box<dyn AdmissionSession + '_> {
    /// Borrows the underlying admission session without adding a second trait-object wrapper.
    fn session(&self) -> &dyn AdmissionSession {
        self.as_ref()
    }
}

/// A query-scoped representation binds the three stages. Associated session
/// types keep native kernels on the stack; existing quantized stage factories
/// retain their boxed sessions. All borrows end before the prepared row drops.
pub(crate) trait QueryPlan: Sync {
    type Prepared<'a>
    where
        Self: 'a;
    type Admission<'a>: SessionHandle
    where
        Self: 'a;

    /// Checks that the selected stages can share one base shape and precision ordering.
    fn validate(&self) -> Result<()>;

    /// Validates one logical query and materializes only the representations its stages need.
    fn prepare<'a>(
        &'a self,
        query: crate::search::query::QueryInput<'a>,
    ) -> Result<Self::Prepared<'a>>;
    // The prepared row may borrow the original input for 'q, but a session
    // only borrows that row for the shorter execution scope 'a.
    /// Borrows an admission session whose lifetime ends before its prepared query storage.
    fn admission<'a, 'q: 'a>(&'a self, query: &'a Self::Prepared<'q>) -> Self::Admission<'a>
    where
        Self: 'q;

    /// Opens the filtering session, including a no-op session for disabled filtering.
    fn prefilter<'a, 'q: 'a>(
        &'a self,
        _query: &'a Self::Prepared<'q>,
    ) -> Box<dyn PrefilterSession + 'a>
    where
        Self: 'q;

    /// Selects final neighbors and optional original-metric scores from completed search scratch.
    fn finish<'a>(
        &'a self,
        query: &Self::Prepared<'a>,
        scratch: &mut InMemSearchScratch,
        k: usize,
        factor: usize,
        scores: Option<ScoreMetric>,
    ) -> SearchRow;
}

impl<D: IndexData + Sync> CompressedIndex<D> {
    /// Validates a plan once and executes independent rows in input order.
    ///
    /// Small batches remain serial; larger batches use Rayon. Every row shares the
    /// same preparation, traversal and result assembly contract.
    pub(crate) fn execute_batch<P: QueryPlan>(
        &self,
        queries: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        plan: &P,
        scores: Option<ScoreMetric>,
    ) -> Result<SearchResult> {
        self.validate_plan(config, plan)?;
        let dimension = self.dataset.dimension();
        if !queries.len().is_multiple_of(dimension) {
            return Err(crate::Error::packed_shape(
                "queries",
                dimension,
                queries.len(),
            ));
        }
        let execute = |row| {
            let query = queries.slice(row * dimension..(row + 1) * dimension);
            self.execute_row(
                query,
                config,
                plan,
                scores,
                NeighborMode::default(),
                &mut NoopObserver,
            )
        };
        let query_count = queries.len() / dimension;
        // Runtime may bypass its pool for a small batch. Respect its explicit
        // limit here too: using a parallel iterator would enter the global pool.
        // Both schedules share the same per-row kernels and output assembly.
        let rows = if query_count <= config.serial_batch_limit.max(1) {
            (0..query_count).map(execute).collect::<Result<Vec<_>>>()?
        } else {
            (0..query_count)
                .into_par_iter()
                .map(execute)
                .collect::<Result<Vec<_>>>()?
        };
        crate::search::output::assemble(rows, config.k, self.num_nodes, scores)
    }

    /// Runs one validated query with caller-selected diagnostic traversal and observer hooks.
    pub(crate) fn execute_observed<P: QueryPlan, O: SearchObserver>(
        &self,
        query: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        plan: &P,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.validate_plan(config, plan)?;
        if query.len() != self.dataset.dimension() {
            return Err(crate::Error::dimension(
                "queries.dimension",
                self.dataset.dimension(),
                query.len(),
            ));
        }
        self.execute_row(query, config, plan, None, mode, observer)
            .map(|row| row.0)
    }

    /// Checks graph/base compatibility and request controls before constructing query sessions.
    fn validate_plan<P: QueryPlan>(&self, config: DynamicSearchConfig, plan: &P) -> Result<()> {
        self.validate_dataset_graph()?;
        config.validate(self.num_nodes)?;
        plan.validate()
    }

    // The observer remains generic so production hooks compile away. The
    // existing search_sessions implementation owns scratch and the hot loop.
    /// Prepares one query and its stages, then borrows scratch for the shared traversal.
    fn execute_row<P: QueryPlan, O: SearchObserver>(
        &self,
        query: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        plan: &P,
        scores: Option<ScoreMetric>,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<SearchRow> {
        let started = SetupTimer::now();
        let query = plan.prepare(query)?;
        let prefilter = plan.prefilter(&query);
        let admission = plan.admission(&query);
        self.search_sessions(
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            prefilter.as_ref(),
            admission.session(),
            mode,
            observer,
            started,
            |scratch, factor| plan.finish(&query, scratch, config.k, factor, scores),
        )
    }
}
