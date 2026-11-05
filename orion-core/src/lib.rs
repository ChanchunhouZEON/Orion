/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![doc = include_str!("../README.md")]
#![cfg_attr(
    not(feature = "benchmark"),
    doc = "
Experiment counters and ablation controls are available with the `benchmark` feature.
```compile_fail
use orion_ann_core::algorithm::search::RAW_VISIT_COUNT;
```
```compile_fail
use orion_ann_core::algorithm::search::set_include_extras;
```

Legacy search signatures also require `benchmark`; use the validated cascade
and native-L2 methods in applications.
```compile_fail,E0599
use orion_ann_core::{StaticOrion, DynamicCascade};
fn legacy(index: &StaticOrion<32>, q: &[f32; 32], recipe: DynamicCascade<'_>) {
    let _ = index.search_unified(q, 1, 8, 3, 0.0, usize::MAX,
        recipe.prefilter, recipe.admission, recipe.rerank);
}
```
```compile_fail,E0599
use orion_ann_core::{StaticOrion, DynamicCascade};
fn legacy(index: &StaticOrion<32>, q: &[[f32; 32]], recipe: DynamicCascade<'_>) {
    let _ = index.search_batch_unified(q, 1, 8, 3, 0.0, usize::MAX,
        recipe.prefilter, recipe.admission, recipe.rerank);
}
```
```compile_fail,E0599
use orion_ann_core::{Orion, DynamicCascade};
use orion_ann_core::algorithm::search::diagnostics::{NeighborMode, NoopObserver};
fn legacy(index: &Orion, q: &[f32], recipe: DynamicCascade<'_>) {
    let _ = index.search_unified_observed(q, 1, 8, 3, 0.0, usize::MAX,
        recipe.prefilter, recipe.admission, recipe.rerank,
        NeighborMode::default(), &mut NoopObserver);
}
```
"
)]
#![warn(missing_debug_implementations)]

// Retain both public import paths while the implementation has a single owner.
pub use orion_ann_error as error;
pub use orion_ann_error::{Error, ErrorCode, ErrorReport, Result};

mod search_result;
pub use search_result::SearchResult;

pub mod algorithm;
pub mod build;
pub use build::{BuildBackend, BuildConfig, BuildMetric};
pub use orion_ann_diskann::model::PaddedDataset;
pub use orion_ann_vector::PaddedLayout;
pub mod index;
pub mod model;
mod pq;
pub mod storage;
pub mod utils;

#[cfg(feature = "benchmark")]
pub use algorithm::SearchProfile;
#[cfg(feature = "benchmark")]
pub use algorithm::search::calibrate::CalibrationDiagnostics;
pub use algorithm::search::calibrate::{CalibratedParams, CalibrationConfig, CalibrationMetric};
pub use algorithm::search::convergence::SearchConvergenceChecker;
#[cfg(feature = "benchmark")]
pub use algorithm::search::diagnostics::SearchProfileStats;
#[cfg(feature = "benchmark")]
pub use index::NaiveOrion;
pub use index::compressed_index::graph::GraphView;
#[cfg(feature = "benchmark")]
pub use index::{BeamSearchConfig, BeamSearchResult, IoLimiter, async_beam_search};
pub use index::{CompressedIndex, DynamicCascade, DynamicSearchConfig, Orion, StaticOrion};
#[cfg(feature = "benchmark")]
pub use index::{DIM_32, DIM_100, DIM_128, DIM_768, DIM_784, DIM_960, DIM_1536};
pub use index::{DiskANNBuildResult, build_diskann_index};
pub(crate) use model::PhasedGraph;
pub use model::{FixedChunkPQTable, NUM_PQ_CENTROIDS};

// Official sidecar contracts; downstream implementations remain sealed.
pub use model::dataset::{
    PreparedQuery, QuantSpec, QuantizedDataset, QuantizedDatasetCompute, QuantizedDatasetShape,
    ResidentMemory,
};
