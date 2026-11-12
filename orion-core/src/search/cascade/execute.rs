/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! One executor for quantized and identity representations. Coordinate/metric
//! dispatch happens before this layer; neither scheduling nor beam traversal
//! inspects the base element type.
use crate::index::IndexData;
use crate::model::scratch::InMemSearchScratch;
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
pub(super) trait SessionHandle {
    fn session(&self) -> &dyn AdmissionSession;
}
impl<T: AdmissionSession> SessionHandle for T {
    fn session(&self) -> &dyn AdmissionSession {
        self
    }
}
impl SessionHandle for Box<dyn AdmissionSession + '_> {
    fn session(&self) -> &dyn AdmissionSession {
        self.as_ref()
    }
}

/// A query-scoped representation binds the three stages. Associated session
/// types keep native kernels on the stack; existing quantized stage factories
/// retain their boxed sessions. All borrows end before the prepared row drops.
pub(super) trait QueryPlan: Sync {
    type Prepared<'a>
    where
        Self: 'a;
    type Admission<'a>: SessionHandle
    where
        Self: 'a;
    fn validate(&self) -> Result<()>;
    fn prepare<'a>(
        &'a self,
        query: crate::search::query::QueryInput<'a>,
    ) -> Result<Self::Prepared<'a>>;
    // The prepared row may borrow the original input for 'q, but a session
    // only borrows that row for the shorter execution scope 'a.
    fn admission<'a, 'q: 'a>(&'a self, query: &'a Self::Prepared<'q>) -> Self::Admission<'a>
    where
        Self: 'q;
    fn prefilter<'a, 'q: 'a>(
        &'a self,
        _query: &'a Self::Prepared<'q>,
    ) -> Box<dyn PrefilterSession + 'a>
    where
        Self: 'q;
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
    pub(super) fn execute_batch<P: QueryPlan>(
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
        let rows = (0..queries.len() / dimension)
            .into_par_iter()
            .map(|row| {
                let query = queries.slice(row * dimension..(row + 1) * dimension);
                self.execute_row(
                    query,
                    config,
                    plan,
                    scores,
                    NeighborMode::default(),
                    &mut NoopObserver,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        crate::search::output::assemble(rows, config.k, self.num_nodes, scores)
    }

    pub(super) fn execute_observed<P: QueryPlan, O: SearchObserver>(
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

    fn validate_plan<P: QueryPlan>(&self, config: DynamicSearchConfig, plan: &P) -> Result<()> {
        self.validate_dataset_graph()?;
        config.validate(self.num_nodes)?;
        plan.validate()
    }

    // The observer remains generic so production hooks compile away. The
    // existing search_sessions implementation owns scratch and the hot loop.
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
