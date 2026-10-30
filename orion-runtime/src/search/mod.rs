/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Representation-specific index and search backends.
//!
//! Timing, calibration, query streaming, and reporting remain in the shared
//! driver. This module only provides operations that depend on the concrete
//! search backend and its resident element representation.

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
/// - loading the corresponding [`orion::Orion`] index,
/// - pinning backend-specific search sidecars,
/// - dispatching the corresponding batch-search pipeline.
pub trait SearchBackend {
    /// Resident element representation used by this backend.
    type Element: BaseElement;

    /// Loads or constructs the index used by this backend.
    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion::Orion<Self::Element>, String>;

    /// Pins backend-specific search storage before measured execution.
    ///
    /// This hook may be a no-op when all required storage is already covered
    /// by the common driver.
    fn pin_search_storage(index: &orion::Orion<Self::Element>, config: &ResolvedRunConfig);

    /// Executes one batch using the backend-specific search pipeline.
    fn search_batch(
        index: &orion::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>>;
}

/// Backend using the conventional `f32` resident representation together with
/// the configured cascade pipeline.
pub struct F32CascadeBackend;

impl SearchBackend for F32CascadeBackend {
    type Element = f32;

    fn load_index(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion::Orion<Self::Element>, String> {
        crate::graph::load_index(config, data)
    }

    fn pin_search_storage(index: &orion::Orion<Self::Element>, config: &ResolvedRunConfig) {
        let cascade = config.search_plan().cascade();

        cascade::pin_cascade(index, cascade.prefilter, cascade.admission, cascade.rerank);
    }

    fn search_batch(
        index: &orion::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
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
    ) -> Result<orion::Orion<Self::Element>, String> {
        crate::graph::load_u8_index(config, data)
    }

    fn pin_search_storage(_index: &orion::Orion<Self::Element>, _config: &ResolvedRunConfig) {
        // No backend-specific sidecar. The resident byte base is pinned by the
        // common driver.
    }

    fn search_batch(
        index: &orion::Orion<Self::Element>,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
        index.search_batch_l2_byte_queries(
            queries,
            orion::DynamicSearchConfig {
                k: config.sweep.k,
                search_list_size: beam,
                window_size: config.window_size,
                epsilon: calibration.threshold,
                early_exit_limit: calibration.early_exit_limit,
            },
        )
    }
}
