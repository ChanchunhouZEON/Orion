/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Command-line presets, input resolution, and experiment controls.
mod cli;
pub(crate) mod pinning;

pub(crate) mod presets;
pub use self::cli::Args;
pub use crate::config::ResolvedRunConfig;
pub use crate::data::{
    query::{open_queries, open_snapshot_queries, GroundTruthReader, QuerySource},
    LoadedBase,
};
pub mod resources {
    pub use crate::preflight::{enforce_budget, preflight_report, print_report};
}
pub use crate::search::{F32CascadeBackend, NativeU8L2Backend, SearchBackend};
