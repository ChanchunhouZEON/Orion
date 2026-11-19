/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Experimental graph-only explicit storage. The graph is read on demand;
//! base and official stage sessions retain their existing resident/mapped paths.
//! This driver never retains a second full graph and has no Tokio dependency.
//! The native slab layout is unchanged; it is not a portable SSD file format.
//!
//! Suspension currently covers graph records only. Base rows, prefilter data,
//! admission codes and rerank data are still borrowed by synchronous sessions.
//! Mapping these objects reduces mandatory residency but does not turn their
//! page faults into async reads. ObjectReader is general enough for later
//! vector/sidecar adapters; those adapters are not implemented by this driver.

use crate::model::graph::storage_graph::StorageGraph;
use crate::model::{
    dataset::rows::{CascadeData, RowTransform},
    search::scratch::InMemSearchScratch,
};
use crate::search::{
    cascade::{
        execute::{QueryPlan, SessionHandle},
        plan::CascadePlan,
    },
    diagnostics::{NeighborMode, NoopObserver, SearchObserver, metrics::SetupTimer},
    query::QueryInput,
};
use crate::{DynamicCascade, DynamicSearchConfig, Result, ScoreMetric, SearchResult};
use orion_ann_platform::reading::ObjectReader;

/// Query execution over a validated object-backed graph.
///
/// The graph owns its reader and layout; each search future owns its prepared
/// sessions and scratch. Multiple queries may borrow the same driver safely.
/// This is an experimental core executor, not the runtime's loading, placement,
/// authentication or calibration policy. Its public surface is available only
/// with the core `experimental-storage` feature.
#[derive(Debug)]
pub struct SearchDriver<R> {
    graph: StorageGraph<R>,
}

impl<R: ObjectReader> SearchDriver<R> {
    /// Attach execution to a prepared graph without copying or reading payloads.
    /// Ownership moves into the driver; reader caches remain shared with any
    /// reader clones retained by the caller.
    pub fn new(graph: StorageGraph<R>) -> Self {
        Self { graph }
    }

    /// Convenience preparation: validate the graph model, then attach execution.
    ///
    /// # Errors
    /// Propagates header reads and graph layout validation failures. The caller
    /// supplies the entry point from the verified immutable generation manifest.
    pub async fn open(reader: R, entry: u32) -> Result<Self> {
        Ok(Self::new(StorageGraph::open(reader, entry).await?))
    }

    /// Inspect the graph model without exposing mutable access to its reader.
    pub fn graph(&self) -> &StorageGraph<R> {
        &self.graph
    }

    /// Search one row with the same query preparation, traversal and score
    /// assembly as the in-memory core API. Controls must already be calibrated
    /// by the application when adaptive search is requested.
    ///
    /// The transform must match how supplied sidecars were fitted. Like the core
    /// cascade API this is an experimental lower-level interface, not runtime's
    /// snapshot/authentication or calibration-policy boundary.
    ///
    /// `base` and the stage datasets must describe the same row IDs and logical
    /// dimension as the graph. `config` supplies search effort and previously
    /// calibrated adaptive controls. `scores` requests exact original-metric
    /// scores for returned IDs; it does not promise exact global top-k.
    ///
    /// # Cancellation
    /// Dropping the future releases its sessions, scratch and acquired leases.
    /// A backend may finish already-submitted reads or populate a shared cache;
    /// cancellation does not imply that the underlying OS operation stopped.
    ///
    /// # Errors
    /// Rejects invalid query/configuration/cascade shapes and incompatible base
    /// counts. Propagates graph-record validation, I/O and resource-limit errors.
    #[allow(clippy::too_many_arguments)]
    pub async fn search<D: CascadeData>(
        &self,
        base: &D,
        query: QueryInput<'_>,
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        transform: RowTransform,
        scores: Option<ScoreMetric>,
    ) -> Result<SearchResult> {
        self.search_observed(
            base,
            query,
            config,
            recipe,
            transform,
            scores,
            &mut NoopObserver,
        )
        .await
    }

    /// Run [`Self::search`] with per-query traversal observations.
    ///
    /// The observer is mutably borrowed for this future's lifetime, including
    /// suspension. Use a separate observer per concurrent query. Callbacks
    /// describe traversal events; they are not a measurement of backend wait
    /// time. Search semantics, cancellation and errors match [`Self::search`].
    #[allow(clippy::too_many_arguments)]
    pub async fn search_observed<D: CascadeData, O: SearchObserver>(
        &self,
        base: &D,
        raw: QueryInput<'_>,
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        transform: RowTransform,
        scores: Option<ScoreMetric>,
        observer: &mut O,
    ) -> Result<SearchResult> {
        let nodes = self.graph.num_nodes();
        config.validate(nodes)?;
        if base.point_count() != nodes || base.len() != nodes {
            return Err(crate::Error::invalid_argument(
                "storage graph/base shape mismatch",
            ));
        }
        let dimension = crate::model::dataset::rows::Rows::dimension(base);
        if raw.len() != dimension {
            return Err(crate::Error::dimension("query", dimension, raw.len()));
        }
        // Prepare the same synchronous compute sessions as the resident path.
        // Only graph acquisition below can yield through ObjectReader.
        let plan = CascadePlan::new(base, recipe, transform);
        plan.validate()?;
        let started = SetupTimer::now();
        let query = plan.prepare(raw)?;
        let filter = plan.prefilter(&query);
        let admission = plan.admission(&query);
        // A suspendable query owns scratch. It must never spin waiting for a
        // resident pool guard while all executor threads are awaiting reads.
        let mut scratch = InMemSearchScratch::new(config.search_list_size);
        let bound = self
            .graph
            .stride_words()
            .checked_mul(4)
            .ok_or_else(|| crate::Error::invalid_data("graph scratch bound overflow"))?;
        scratch.dist_buffer.try_reserve(bound).map_err(|error| {
            crate::Error::invalid_data(format!("graph scratch allocation failed: {error}"))
        })?;
        // Scratch belongs to this future across waits; no pool slot is held
        // while another query needs the same executor to complete its I/O.
        let result = super::traversal::traverse(
            &self.graph,
            self.graph.entry_point(),
            &mut scratch,
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            filter.as_ref(),
            admission.session(),
            NeighborMode::default(),
            observer,
            started,
            |scratch, factor| plan.finish(&query, scratch, config.k, factor, scores),
        )
        .await?;
        super::output::assemble(vec![result], config.k, nodes, scores)
    }
}

#[cfg(test)]
mod tests;
