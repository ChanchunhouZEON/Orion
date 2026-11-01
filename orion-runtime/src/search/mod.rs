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
use orion_ann::algorithm::search::diagnostics::SearchObserver;
pub mod cascade;
pub mod utils;

use crate::{
    config::ResolvedRunConfig,
    data::{BaseElement, LoadedBase},
};

/// Backend-specific hooks used by the common search driver.
///
/// Each backend selects exactly one resident element representation through
/// [`SearchBackend::Element`]. The common driver owns timing, calibration,
/// query streaming, and reporting, while implementations provide only the
/// operations that vary by backend:
///
/// - loading the corresponding [`orion_ann::Orion`] index,
/// - pinning backend-specific search sidecars,
/// - dispatching the corresponding batch-search pipeline.
pub trait SearchBackend {
    /// Resident element representation used by this backend.
    type Element: BaseElement;

    /// Loads or constructs the index used by this backend.
    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann::Orion<Self::Element>>;

    /// Pins backend-specific search storage before measured execution.
    ///
    /// This hook may be a no-op when all required storage is already covered
    /// by the common driver.
    fn pin_search_storage(index: &orion_ann::Orion<Self::Element>, config: &ResolvedRunConfig);

    /// Executes one batch using the backend-specific search pipeline.
    fn search_batch(
        index: &orion_ann::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion_ann::CalibratedParams,
    ) -> Result<Vec<Vec<u32>>>;

    /// Optional diagnostic single-query path; third-party backends may opt in.
    fn search_observed<O: SearchObserver>(
        _index: &orion_ann::Orion<Self::Element>,
        _query: &[f32],
        _config: &ResolvedRunConfig,
        _beam: usize,
        _calibration: orion_ann::CalibratedParams,
        _observer: &mut O,
    ) -> Result<Vec<u32>> {
        Err(crate::Error::unsupported(
            "search backend does not provide diagnostic traces",
        ))
    }
}

/// Backend using the conventional `f32` resident representation together with
/// the configured cascade pipeline.
pub struct F32CascadeBackend;

impl SearchBackend for F32CascadeBackend {
    type Element = f32;

    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann::Orion<Self::Element>> {
        crate::graph::load_index(config, data)
    }

    fn pin_search_storage(index: &orion_ann::Orion<Self::Element>, config: &ResolvedRunConfig) {
        let cascade = config.search_plan().cascade();

        cascade::pin_cascade(index, cascade.prefilter, cascade.admission, cascade.rerank);
    }

    fn search_batch(
        index: &orion_ann::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion_ann::CalibratedParams,
    ) -> Result<Vec<Vec<u32>>> {
        let cascade = config.search_plan().cascade();

        cascade::search_batch_compose(
            index,
            queries,
            config.sweep.k,
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
        index: &orion_ann::Orion<Self::Element>,
        query: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion_ann::CalibratedParams,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        let stages = config.search_plan().cascade();
        let ad = cascade::build_admission(index, stages.admission)?;
        let pf = cascade::build_prefilter(index, stages.prefilter, stages.admission);
        let rr = cascade::build_rerank(index, stages.rerank);
        index.search_cascade_observed(
            query,
            diagnostic_config(config, beam, calibration),
            orion_ann::DynamicCascade {
                prefilter: pf.as_deref(),
                admission: ad.as_ref(),
                rerank: rr.as_ref(),
            },
            if orion_ann::algorithm::search::include_extras() {
                orion_ann::algorithm::search::diagnostics::NeighborMode::LocalExtra
            } else {
                orion_ann::algorithm::search::diagnostics::NeighborMode::FullNeighbor
            },
            observer,
        )
    }
}

/// Backend using native `u8` resident storage.
///
/// Exact L2 admission operates directly on the resident byte vectors, so this
/// backend requires neither a quantized admission sidecar nor an `f32` rerank
/// stage.
pub struct NativeU8L2Backend;

impl SearchBackend for NativeU8L2Backend {
    type Element = u8;

    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion_ann::Orion<Self::Element>> {
        crate::graph::load_u8_index(config, data)
    }

    fn pin_search_storage(_index: &orion_ann::Orion<Self::Element>, _config: &ResolvedRunConfig) {
        // No backend-specific sidecar. The resident byte base is pinned by the
        // common driver.
    }

    fn search_batch(
        index: &orion_ann::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion_ann::CalibratedParams,
    ) -> Result<Vec<Vec<u32>>> {
        index.search_batch_l2_byte_queries(
            queries,
            orion_ann::DynamicSearchConfig {
                k: config.sweep.k,
                search_list_size: beam,
                window_size: config.window_size,
                epsilon: calibration.threshold,
                early_exit_limit: calibration.early_exit_limit,
            },
        )
    }
    fn search_observed<O: SearchObserver>(
        index: &orion_ann::Orion<Self::Element>,
        query: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion_ann::CalibratedParams,
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
            diagnostic_config(config, beam, calibration),
            observer,
        )
    }
}

fn diagnostic_config(
    config: &ResolvedRunConfig,
    beam: usize,
    calibration: orion_ann::CalibratedParams,
) -> orion_ann::DynamicSearchConfig {
    orion_ann::DynamicSearchConfig {
        k: config.sweep.k,
        search_list_size: beam,
        window_size: config.window_size,
        epsilon: calibration.threshold,
        early_exit_limit: calibration.early_exit_limit,
    }
}
