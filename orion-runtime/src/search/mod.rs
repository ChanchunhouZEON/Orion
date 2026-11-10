/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Representation-specific index and search backends.
//!
//! Timing, calibration, query streaming, and reporting remain in the shared
//! driver. This module only provides operations that depend on the concrete
//! search backend and its resident element representation.

use crate::Result;
use orion_ann_core::algorithm::search::diagnostics::SearchObserver;
pub mod cascade;
pub mod utils;

use crate::{config::ResolvedRunConfig, data::BaseElement};

#[cfg(feature = "cli")]
use crate::data::LoadedBase;

mod sealed {
    pub trait BuiltIn {}
    impl BuiltIn for super::F32CascadeBackend {}
    impl BuiltIn for super::NativeU8L2Backend {}
}

/// Backend-specific hooks used by the common search driver.
///
/// Each backend selects exactly one resident element representation through
/// [`SearchBackend::Element`]. The common driver owns timing, calibration,
/// query streaming, and reporting, while implementations provide only the
/// operations that vary by backend:
///
/// - loading the corresponding [`orion_ann_core::Orion`] index,
/// - pinning backend-specific search sidecars,
/// - dispatching the corresponding batch-search pipeline.
pub trait SearchBackend: sealed::BuiltIn {
    /// Resident element representation used by this backend.
    type Element: BaseElement;

    fn search_batch_scored(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult>;

    fn search_batch_u8_scored(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[u8],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        let values: Vec<f32> = queries.iter().map(|&v| f32::from(v)).collect();
        Self::search_batch_scored(index, &values, config, k, beam, calibration)
    }

    /// Loads or constructs the index used by this backend.
    #[cfg(feature = "cli")]
    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann_core::Orion<Self::Element>>;

    /// Lock already-prepared requirements. The guard must outlive measured work.
    fn pin_search_storage<'a>(
        index: &'a orion_ann_core::Orion<Self::Element>,
        config: &ResolvedRunConfig,
        options: &crate::PinOptions,
    ) -> Result<crate::PinnedDatasets<'a>> {
        crate::residency::pin(
            index,
            bytemuck::cast_slice(index.dataset().as_slice()),
            orion_ann_core::QuantizedDatasetShape {
                rows: index.len(),
                dimension: index.dataset().layout().logical_dim(),
            },
            &config.cascade().dataset_requirements(),
            options,
        )
    }

    fn pin_hints(
        index: &orion_ann_core::Orion<Self::Element>,
        config: &ResolvedRunConfig,
    ) -> Result<Vec<crate::PinHint>> {
        crate::residency::hints(
            index,
            bytemuck::cast_slice(index.dataset().as_slice()),
            orion_ann_core::QuantizedDatasetShape {
                rows: index.len(),
                dimension: index.dataset().layout().logical_dim(),
            },
            &config.cascade().dataset_requirements(),
        )
    }

    /// Prepare a search generation, including its optional automatic residency.
    /// All frontends retain the returned guard for the search lifetime. This is
    /// independent of the `cli` feature and never runs in the per-query path.
    fn prepare_search<'a>(
        index: &'a orion_ann_core::Orion<Self::Element>,
        config: &ResolvedRunConfig,
    ) -> Result<crate::PinnedDatasets<'a>> {
        crate::placement::configure(index, config, std::iter::empty())?;
        // Placement validates the live generation before its first allocation.
        // Replanning from current free memory after sidecars exist would count
        // them twice and could report a placement different from their backing.
        Self::prepare_datasets(index, config)?;
        crate::residency::automatic_pin(
            index,
            bytemuck::cast_slice(index.dataset().as_slice()),
            orion_ann_core::QuantizedDatasetShape {
                rows: index.len(),
                dimension: index.dataset().layout().logical_dim(),
            },
            &config.cascade().dataset_requirements(),
            config.auto_pin,
            config.memory_budget_gib,
        )
    }

    /// Representation-specific materialization, before the common residency step.
    fn prepare_datasets(
        index: &orion_ann_core::Orion<Self::Element>,
        config: &ResolvedRunConfig,
    ) -> Result<()>;

    /// Executes one batch using the backend-specific search pipeline.
    fn search_batch(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult>;

    /// Byte input for native storage; the f32 backend converts before dispatch.
    fn search_batch_u8(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[u8],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        let values: Vec<f32> = queries.iter().map(|&v| f32::from(v)).collect();
        Self::search_batch(index, &values, config, k, beam, calibration)
    }

    /// Optional diagnostic single-query path for built-in backends.
    fn search_observed<O: SearchObserver>(
        _index: &orion_ann_core::Orion<Self::Element>,
        _query: &[f32],
        _config: &ResolvedRunConfig,
        _k: usize,
        _beam: usize,
        _calibration: orion_ann_core::CalibratedParams,
        _observer: &mut O,
    ) -> Result<Vec<u32>> {
        Err(crate::Error::unsupported(
            "search backend does not provide diagnostic traces",
        ))
    }
}

/// Backend using the conventional `f32` resident representation together with
/// the configured cascade pipeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct F32CascadeBackend;

impl SearchBackend for F32CascadeBackend {
    type Element = f32;

    fn search_batch_scored(
        index: &orion_ann_core::Orion<f32>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        let stages = config.cascade();
        let ad = cascade::build_admission(index, stages.admission)?;
        let pf = cascade::build_prefilter(index, stages.prefilter, stages.admission)?;
        let rr = cascade::build_rerank(index, stages.rerank)?;
        let metric = match config.metric {
            crate::SearchMetric::L2 => orion_ann_core::ScoreMetric::SquaredL2,
            crate::SearchMetric::InnerProduct => orion_ann_core::ScoreMetric::InnerProduct,
            crate::SearchMetric::Cosine => orion_ann_core::ScoreMetric::CosineSimilarity,
        };
        index.search_batch_cascade_scored(
            queries,
            search_config(config, k, beam, calibration),
            orion_ann_core::DynamicCascade {
                prefilter: pf.as_deref(),
                admission: ad.as_ref(),
                rerank: rr.as_ref(),
            },
            metric,
        )
    }

    fn prepare_datasets(
        index: &orion_ann_core::Orion<f32>,
        config: &ResolvedRunConfig,
    ) -> Result<()> {
        crate::placement::configure(index, config, std::iter::empty())?;
        crate::sidecars::requirements::prepare(
            index,
            &config.index_plan().sidecar_requirements(false),
        )
    }

    #[cfg(feature = "cli")]
    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann_core::Orion<Self::Element>> {
        crate::graph::load_index(config, data)
    }

    fn search_batch(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        let cascade = config.cascade();

        cascade::search_batch_compose(
            index,
            queries,
            k,
            beam,
            config.window_size,
            calibration.threshold,
            calibration.early_exit_limit,
            cascade.prefilter,
            cascade.admission,
            cascade.rerank,
        )
    }
    fn search_observed<O: SearchObserver>(
        index: &orion_ann_core::Orion<Self::Element>,
        query: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        crate::placement::configure(index, config, std::iter::empty())?;
        crate::sidecars::requirements::prepare(
            index,
            &config.index_plan().sidecar_requirements(false),
        )?;
        let stages = config.cascade();
        let ad = cascade::build_admission(index, stages.admission)?;
        let pf = cascade::build_prefilter(index, stages.prefilter, stages.admission)?;
        let rr = cascade::build_rerank(index, stages.rerank)?;
        index.search_cascade_observed(
            query,
            search_config(config, k, beam, calibration),
            orion_ann_core::DynamicCascade {
                prefilter: pf.as_deref(),
                admission: ad.as_ref(),
                rerank: rr.as_ref(),
            },
            orion_ann_core::algorithm::search::diagnostics::NeighborMode::default(),
            observer,
        )
    }
}

/// Backend using native `u8` resident storage.
///
/// Exact L2 admission operates directly on the resident byte vectors, so this
/// backend requires neither a quantized admission sidecar nor an `f32` rerank
/// stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeU8L2Backend;

impl SearchBackend for NativeU8L2Backend {
    type Element = u8;

    fn search_batch_scored(
        index: &orion_ann_core::Orion<u8>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        index.search_batch_l2_byte_queries_scored(
            queries,
            search_config(config, k, beam, calibration),
        )
    }

    fn search_batch_u8_scored(
        index: &orion_ann_core::Orion<u8>,
        queries: &[u8],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        index.search_batch_l2_scored(queries, search_config(config, k, beam, calibration))
    }

    fn prepare_datasets(
        _index: &orion_ann_core::Orion<u8>,
        _config: &ResolvedRunConfig,
    ) -> Result<()> {
        Ok(()) // Native admission shares the base and needs no sidecars.
    }

    fn search_batch_u8(
        index: &orion_ann_core::Orion<u8>,
        queries: &[u8],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        index.search_batch_l2(queries, search_config(config, k, beam, calibration))
    }

    #[cfg(feature = "cli")]
    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann_core::Orion<Self::Element>> {
        crate::graph::load_u8_index(config, data)
    }

    fn search_batch(
        index: &orion_ann_core::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        index.search_batch_l2_byte_queries(
            queries,
            orion_ann_core::DynamicSearchConfig {
                k,
                search_list_size: beam,
                window_size: config.window_size,
                epsilon: calibration.threshold,
                early_exit_limit: calibration.early_exit_limit,
            },
        )
    }
    fn search_observed<O: SearchObserver>(
        index: &orion_ann_core::Orion<Self::Element>,
        query: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        calibration: orion_ann_core::CalibratedParams,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        if query
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || *v > 255. || v.fract() != 0.)
        {
            return Err(crate::Error::invalid_argument(
                "native u8 query must contain exact byte values",
            ));
        }
        let bytes: Vec<u8> = query.iter().map(|&v| v as u8).collect();
        index.search_l2_observed(
            &bytes,
            search_config(config, k, beam, calibration),
            observer,
        )
    }
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
