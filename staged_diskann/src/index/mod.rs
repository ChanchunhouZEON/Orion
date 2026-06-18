pub mod builder;
pub mod compressed_index;
pub mod naive_compressed_index;

pub use crate::algorithm::search::async_beam_search::{
    BeamSearchConfig, BeamSearchResult, IoLimiter, async_beam_search,
};
pub use builder::{DiskANNBuildResult, build_diskann_index};
pub use compressed_index::StagedDiskANN;
pub use naive_compressed_index::NaiveStagedDiskANN;

/// Supported dimension constants for dispatch.
pub const DIM_32: usize = 32;
pub const DIM_100: usize = 100;
pub const DIM_128: usize = 128;
pub const DIM_768: usize = 768;
pub const DIM_784: usize = 784;
pub const DIM_960: usize = 960;
pub const DIM_1536: usize = 1536;
