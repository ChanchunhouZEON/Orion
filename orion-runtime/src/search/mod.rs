/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime adapters for validated configuration and resident representations.
//!
//! The core search module owns query/session preparation and a single executor.
//! These adapters only bind runtime options to core stages; timing, persistence,
//! calibration and residency remain outside query execution.

use crate::{config::ResolvedRunConfig, data::BaseElement, Result};
use orion_ann_core::{
    model::dataset::rows::CascadeData,
    search::{diagnostics::SearchObserver, query::QueryInput},
    Orion,
};
use orion_ann_diskann::model::PaddedDataset;
pub mod cascade;
pub mod utils;

/// Bind representation semantics before either fitting or snapshot restoration.
pub(crate) fn bind_rows<T: BaseElement>(
    index: &Orion<T>,
    config: &ResolvedRunConfig,
) -> Result<()> {
    use orion_ann_core::model::dataset::rows::RowTransform;
    index.set_sidecar_row_transform(
        if config.storage() == crate::VectorStorageKind::U8
            && config.metric == crate::SearchMetric::Cosine
        {
            RowTransform::UnitL2
        } else {
            RowTransform::Identity
        },
    )
}

pub fn prepare_datasets<T: BaseElement>(index: &Orion<T>, config: &ResolvedRunConfig) -> Result<()>
where
    PaddedDataset<T>: CascadeData,
{
    bind_rows(index, config)?;
    crate::placement::configure(index, config, std::iter::empty())?;
    if config.storage() == crate::VectorStorageKind::U8
        && config.metric == crate::SearchMetric::Cosine
    {
        index.exact_stage(orion_ann_core::ScoreMetric::CosineSimilarity)?;
    }
    crate::sidecars::requirements::prepare(index, &config.index_plan().sidecar_requirements(false))
}
pub fn prepare_search<'a, T: BaseElement>(
    index: &'a Orion<T>,
    config: &ResolvedRunConfig,
) -> Result<crate::PinnedDatasets<'a>>
where
    PaddedDataset<T>: CascadeData,
{
    prepare_datasets(index, config)?;
    crate::residency::automatic_pin(
        index,
        bytemuck::cast_slice(index.dataset().as_slice()),
        shape(index),
        &config.cascade().dataset_requirements(),
        config.auto_pin,
        config.memory_budget_gib,
    )
}
pub fn pin_search_storage<'a, T: BaseElement>(
    index: &'a Orion<T>,
    config: &ResolvedRunConfig,
    options: &crate::PinOptions,
) -> Result<crate::PinnedDatasets<'a>> {
    crate::residency::pin(
        index,
        bytemuck::cast_slice(index.dataset().as_slice()),
        shape(index),
        &config.cascade().dataset_requirements(),
        options,
    )
}
pub fn pin_hints<T: BaseElement>(
    index: &Orion<T>,
    config: &ResolvedRunConfig,
) -> Result<Vec<crate::PinHint>> {
    crate::residency::hints(
        index,
        bytemuck::cast_slice(index.dataset().as_slice()),
        shape(index),
        &config.cascade().dataset_requirements(),
    )
}
fn shape<T: BaseElement>(index: &Orion<T>) -> orion_ann_core::QuantizedDatasetShape {
    orion_ann_core::QuantizedDatasetShape {
        rows: index.len(),
        dimension: index.dataset().layout().logical_dim(),
    }
}

/// Runtime supplies a recipe; core owns element/metric dispatch and scheduling.
pub fn search_batch<T: BaseElement>(
    index: &Orion<T>,
    queries: QueryInput<'_>,
    config: &ResolvedRunConfig,
    k: usize,
    beam: usize,
    calibration: orion_ann_core::CalibratedParams,
    scores: bool,
) -> Result<orion_ann_core::SearchResult>
where
    PaddedDataset<T>: CascadeData,
{
    let stages = config.cascade();
    stages.validate_cascade_options(config.metric)?;
    bind_rows(index, config)?;
    crate::sidecars::requirements::prepare(index, &stages.sidecar_requirements())?;
    let ad = cascade::build_admission(index, stages.admission)?;
    let pf = cascade::build_prefilter(index, stages.prefilter, stages.admission)?;
    let rr = cascade::build_rerank(index, stages.rerank)?;
    index.search_batch(
        queries,
        search_config(config, k, beam, calibration),
        orion_ann_core::DynamicCascade {
            prefilter: pf.as_ref(),
            admission: ad.as_ref(),
            rerank: rr.as_ref(),
        },
        scores.then(|| score_metric(config)),
    )
}
pub fn search_observed<T: BaseElement, O: SearchObserver>(
    index: &Orion<T>,
    query: QueryInput<'_>,
    config: &ResolvedRunConfig,
    k: usize,
    beam: usize,
    calibration: orion_ann_core::CalibratedParams,
    observer: &mut O,
) -> Result<Vec<u32>>
where
    PaddedDataset<T>: CascadeData,
{
    prepare_datasets(index, config)?;
    let stages = config.cascade();
    let ad = cascade::build_admission(index, stages.admission)?;
    let pf = cascade::build_prefilter(index, stages.prefilter, stages.admission)?;
    let rr = cascade::build_rerank(index, stages.rerank)?;
    index.search_observed(
        query,
        search_config(config, k, beam, calibration),
        orion_ann_core::DynamicCascade {
            prefilter: pf.as_ref(),
            admission: ad.as_ref(),
            rerank: rr.as_ref(),
        },
        orion_ann_core::search::diagnostics::NeighborMode::default(),
        observer,
    )
}

fn search_config(
    config: &ResolvedRunConfig,
    k: usize,
    beam: usize,
    calibration: orion_ann_core::CalibratedParams,
) -> orion_ann_core::DynamicSearchConfig {
    orion_ann_core::DynamicSearchConfig {
        k,
        search_list_size: beam,
        window_size: config.window_size,
        epsilon: calibration.threshold,
        early_exit_limit: calibration.early_exit_limit,
    }
}

fn score_metric(config: &ResolvedRunConfig) -> orion_ann_core::ScoreMetric {
    match config.metric {
        crate::SearchMetric::SquaredL2 => orion_ann_core::ScoreMetric::SquaredL2,
        crate::SearchMetric::InnerProduct => orion_ann_core::ScoreMetric::InnerProduct,
        crate::SearchMetric::Cosine => orion_ann_core::ScoreMetric::CosineSimilarity,
    }
}
