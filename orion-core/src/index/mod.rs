/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

pub mod builder;
pub mod compressed_index;
#[cfg(feature = "benchmark")]
pub mod naive_compressed_index;

#[cfg(feature = "benchmark")]
pub use crate::algorithm::search::async_beam_search::{
    BeamSearchConfig, BeamSearchResult, IoLimiter, async_beam_search,
};
pub use builder::{DiskANNBuildResult, build_diskann_index};
pub use compressed_index::{
    CompressedIndex, DynamicCascade, DynamicSearchConfig, Orion, StaticOrion,
};
#[cfg(feature = "benchmark")]
pub use naive_compressed_index::NaiveOrion;

/// Supported dimension constants for dispatch.
#[cfg(feature = "benchmark")]
pub const DIM_32: usize = 32;
#[cfg(feature = "benchmark")]
pub const DIM_100: usize = 100;
#[cfg(feature = "benchmark")]
pub const DIM_128: usize = 128;
#[cfg(feature = "benchmark")]
pub const DIM_768: usize = 768;
#[cfg(feature = "benchmark")]
pub const DIM_784: usize = 784;
#[cfg(feature = "benchmark")]
pub const DIM_960: usize = 960;
#[cfg(feature = "benchmark")]
pub const DIM_1536: usize = 1536;
