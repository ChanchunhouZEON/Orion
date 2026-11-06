/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime-dimensional cascades use the same graph loop as the typed facade.
use super::{CompressedIndex, DynamicSearchConfig};
use crate::algorithm::search::{
    diagnostics::{NeighborMode, NoopObserver, SearchObserver},
    stage::{AdmissionStage, PrefilterStage, RerankStage},
};
use crate::{Error, Result};

use rayon::prelude::*;

/// Borrow a prepared recipe once, then share it across all queries/workers.
/// Factories quantize/project each query once; sessions run the existing SIMD
/// distance streams. No runtime dimension dispatch occurs per coordinate.
#[derive(Clone, Copy)]
pub struct DynamicCascade<'a> {
    pub prefilter: Option<&'a dyn PrefilterStage>,
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
            self.prefilter.and_then(|p| p.precision()),
            admission,
            self.rerank.precision(),
        )?;
        let shapes = [
            self.prefilter.and_then(|p| p.shape()),
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
impl<D: crate::model::dataset::rows::CascadeData> CompressedIndex<D> {
    fn validate_cascade(
        &self,
        queries: &[f32],
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
    ) -> Result<()> {
        self.validate_dataset_graph()?;
        if config.k == 0
            || config.k > self.num_nodes
            || config.search_list_size < config.k
            || !(1..=64).contains(&config.window_size)
            || !config.epsilon.is_finite()
            || !(0.0..=1.0).contains(&config.epsilon)
        {
            return Err(Error::invalid_argument("invalid cascade search controls"));
        }
        let dim = crate::model::dataset::rows::F32Rows::dimension(&self.dataset);
        if queries.len() % dim != 0 || queries.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid_argument("invalid cascade query matrix"));
        }
        recipe.validate(dim, self.num_nodes)
    }
    pub fn search_cascade(
        &self,
        query: &[f32],
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
    ) -> Result<Vec<u32>> {
        self.search_cascade_observed(
            query,
            config,
            recipe,
            self.cascade_mode(),
            &mut NoopObserver,
        )
    }
    pub fn search_batch_cascade(
        &self,
        queries: &[f32],
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
    ) -> Result<crate::SearchResult> {
        self.validate_cascade(queries, config, recipe)?;
        queries
            .par_chunks(crate::model::dataset::rows::F32Rows::dimension(
                &self.dataset,
            ))
            .map(|query| {
                self.run_cascade(
                    query,
                    config,
                    recipe,
                    self.cascade_mode(),
                    &mut NoopObserver,
                )
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|rows| crate::SearchResult::from_rows(rows, config.k, self.num_nodes))
    }
    pub fn search_cascade_observed<O: SearchObserver>(
        &self,
        query: &[f32],
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.validate_cascade(query, config, recipe)?;
        if query.len() != crate::model::dataset::rows::F32Rows::dimension(&self.dataset) {
            return Err(Error::invalid_argument("expected one logical query row"));
        }
        self.run_cascade(query, config, recipe, mode, observer)
    }
    fn cascade_mode(&self) -> NeighborMode {
        NeighborMode::default()
    }
    fn run_cascade<O: SearchObserver>(
        &self,
        query: &[f32],
        config: DynamicSearchConfig,
        recipe: DynamicCascade<'_>,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        let started = crate::algorithm::search::metrics::SetupTimer::now();
        let prefilter = recipe.prefilter.map(|stage| stage.open(query));
        let admission = recipe.admission.open(query);
        self.search_sessions(
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            prefilter.as_deref(),
            admission.as_ref(),
            mode,
            observer,
            started,
            |scratch, factor| recipe.rerank.rerank(query, &[], config.k, factor, scratch),
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
        prefilter: Option<&dyn PrefilterStage>,
        admission: &dyn AdmissionStage,
        rerank: &dyn RerankStage,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.search_cascade_observed(
            query,
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
            .field("has_prefilter", &self.prefilter.is_some())
            .field("admission_shape", &self.admission.shape())
            .field("rerank_shape", &self.rerank.shape())
            .finish_non_exhaustive()
    }
}
