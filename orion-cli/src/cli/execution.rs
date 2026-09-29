/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Representation-specific index and search backends.
//!
//! Timing, calibration, query streaming, and reporting remain in the shared
//! driver. This module only provides operations that depend on the concrete
//! search backend and its resident element representation.

use crate::cascade;
use crate::cli::{
    config::ResolvedRunConfig,
    data::{BaseElement, LoadedBase},
};
use orion::algorithm::search::calibrate::CalibrationElement;

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
    type Element: BaseElement + CalibrationElement;

    /// Loads or constructs the index used by this backend.
    fn load_index<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion::Orion<N, Self::Element>, String>
    where
        [Self::Element; N]:
        vector::FullPrecisionDistance<Self::Element, N>;

    /// Pins backend-specific search storage before measured execution.
    ///
    /// This hook may be a no-op when all required storage is already covered
    /// by the common driver.
    fn pin_search_storage<const N: usize>(
        index: &orion::Orion<N, Self::Element>,
        config: &ResolvedRunConfig,
    )
    where
        [Self::Element; N]:
        vector::FullPrecisionDistance<Self::Element, N>;

    /// Executes one batch using the backend-specific search pipeline.
    fn search_batch<const N: usize>(
        index: &orion::Orion<N, Self::Element>,
        queries: &[[f32; N]],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>>
    where
        [Self::Element; N]:
        vector::FullPrecisionDistance<Self::Element, N>;
}

/// Backend using the conventional `f32` resident representation together with
/// the configured cascade pipeline.
pub struct F32CascadeBackend;

impl SearchBackend for F32CascadeBackend {
    type Element = f32;

    fn load_index<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion::Orion<N, Self::Element>, String> {
        crate::cli::index::load_index(config, data)
    }

    fn pin_search_storage<const N: usize>(
        index: &orion::Orion<N, Self::Element>,
        config: &ResolvedRunConfig,
    ) {
        let cascade = config.search_plan().cascade();

        cascade::pin_cascade(
            index,
            cascade.prefilter,
            cascade.admission,
            cascade.rerank,
        );
    }

    fn search_batch<const N: usize>(
        index: &orion::Orion<N, Self::Element>,
        queries: &[[f32; N]],
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

    fn load_index<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedBase<Self::Element>,
    ) -> Result<orion::Orion<N, Self::Element>, String> {
        crate::cli::index::load_u8_index(config, data)
    }

    fn pin_search_storage<const N: usize>(
        _index: &orion::Orion<N, Self::Element>,
        _config: &ResolvedRunConfig,
    ) {
        // No backend-specific sidecar. The resident byte base is pinned by the
        // common driver.
    }

    fn search_batch<const N: usize>(
        index: &orion::Orion<N, Self::Element>,
        queries: &[[f32; N]],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
        use orion::algorithm::search::stage::{
            admission::NativeU8Admission,
            prefilter::NoPrefilter,
            rerank::NoRerank,
        };

        let admission = NativeU8Admission::new(&index.dataset);

        index.search_batch_unified(
            queries,
            config.sweep.k,
            beam,
            config.window_size,
            calibration.threshold,
            calibration.early_exit_limit,
            None::<&NoPrefilter>,
            &admission,
            &NoRerank,
        )
    }
}