/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Command-line presets, input resolution, and experiment controls.
mod cli;

pub(crate) mod presets;
pub use self::cli::Args;
pub use crate::config::ResolvedRunConfig;
pub use crate::data::{
    query::{open_queries, open_snapshot_queries, GroundTruthReader, QuerySource},
    LoadedBase,
};
pub mod resources {
    pub use crate::preflight::{enforce_budget, preflight_report, print_report};

    /// Hold best-effort RAM locks while the CLI searches a loaded snapshot.
    /// Opening the index already verified payloads and enforced its memory budget.
    /// OS refusals are logged; invalid/unprepared requirements return an error.
    pub fn pin_snapshot_storage(index: &crate::Index) -> crate::Result<crate::PinnedDatasets<'_>> {
        index.pin_for_driver()
    }
}
pub use crate::search::{F32CascadeBackend, NativeU8L2Backend, SearchBackend};
