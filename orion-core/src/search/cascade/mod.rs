/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime-dimensional cascades use the same graph loop as the typed facade.
use crate::index::{CompressedIndex, DynamicSearchConfig};
use crate::search::{
    cascade::stage::{AdmissionStage, PrefilterStage, RerankStage},
    diagnostics::{NeighborMode, SearchObserver},
};
use crate::{Error, Result};

/// Borrow a prepared recipe once, then share it across all queries/workers.
/// Factories quantize/project each query once; sessions run the existing SIMD
/// distance streams. No runtime dimension dispatch occurs per coordinate.
#[derive(Clone, Copy)]
pub struct DynamicCascade<'a> {
    /// Use `NoPrefilter` to send neighbors directly to admission.
    pub prefilter: &'a dyn PrefilterStage,
    pub admission: &'a dyn AdmissionStage,
    pub rerank: &'a dyn RerankStage,
}
impl DynamicCascade<'_> {
    fn validate(&self, dimension: usize, points: usize) -> Result<()> {
        use crate::model::dataset::quantized_dataset::capabilities::validate_precision;
        let admission = self.admission.precision().ok_or_else(|| {
            Error::invalid_argument("admission requires a scoring representation")
        })?;
        validate_precision(
            self.prefilter.precision(),
            admission,
            self.rerank.precision(),
        )?;
        let shapes = [
            self.prefilter.shape(),
            self.admission.shape(),
            self.rerank.shape(),
        ];
        if shapes
            .into_iter()
            .flatten()
            .any(|s| s != (dimension, points))
        {
            return Err(Error::invalid_argument(
                "cascade sidecar shape does not match the base",
            ));
        }
        Ok(())
    }
}
mod plan;
use plan::CascadePlan;
impl<D: crate::model::dataset::rows::CascadeData> CompressedIndex<D> {
    /// Search borrowed packed queries with a prepared, validated cascade.
    /// Optional output scores always use the original metric/base coordinates.
    pub fn search_batch(
        &self,
        queries: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        scores: Option<crate::ScoreMetric>,
    ) -> Result<crate::SearchResult> {
        self.execute_batch(
            queries,
            config,
            &CascadePlan::new(&self.dataset, recipe, self.sidecar_row_transform()),
            scores,
        )
    }

    /// A single-row counterpart with the same score and validation contract.
    pub fn search(
        &self,
        query: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        scores: Option<crate::ScoreMetric>,
    ) -> Result<crate::SearchResult> {
        let dimension = crate::model::dataset::rows::Rows::dimension(&self.dataset);
        if query.len() != dimension {
            return Err(Error::dimension("query", dimension, query.len()));
        }
        self.search_batch(query, config, recipe, scores)
    }
    /// Diagnostics use the same prepared stages and traversal as batch search.
    pub fn search_observed<O: SearchObserver>(
        &self,
        query: crate::search::query::QueryInput<'_>,
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.execute_observed(
            query,
            config,
            &CascadePlan::new(&self.dataset, recipe, self.sidecar_row_transform()),
            mode,
            observer,
        )
    }
}

#[cfg(feature = "benchmark")]
impl CompressedIndex<orion_ann_diskann::model::PaddedDataset<f32>> {
    /// Compatibility call shape for diagnostic observers; queries and storage
    /// are runtime-dimensional and use the same validated cascade entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn search_unified_observed<O: SearchObserver>(
        &self,
        query: &[f32],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: &dyn PrefilterStage,
        admission: &dyn AdmissionStage,
        rerank: &dyn RerankStage,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.search_observed(
            crate::search::query::QueryInput::F32(query),
            DynamicSearchConfig {
                k,
                search_list_size,
                window_size,
                epsilon,
                early_exit_limit,
            },
            DynamicCascade {
                prefilter,
                admission,
                rerank,
            },
            mode,
            observer,
        )
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for DynamicCascade<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicCascade")
            .field("has_prefilter", &self.prefilter.precision().is_some())
            .field("admission_shape", &self.admission.shape())
            .field("rerank_shape", &self.rerank.shape())
            .finish_non_exhaustive()
    }
}

// A recipe is bound once, then executed independently of base element width.
mod execute;
pub mod stage;
