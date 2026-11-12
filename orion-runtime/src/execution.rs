/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Adapter support for prepared runs. No argument parsing, environment or presets.
//! Ordinary Rust applications use IndexOptions and Index directly.
pub use crate::config::{PreflightFormat, ResolvedRunConfig};
pub use crate::data::{
    inspect_vector_file,
    query::{open_queries, open_snapshot_queries, GroundTruthReader, QuerySource},
    LoadedBase, VectorFormat,
};
pub mod resources {
    pub use crate::preflight::{enforce_budget, preflight_report, print_report};
}
pub use crate::search::{prepare_search, search_batch};

/// Resolve validated service options against caller-provided input metadata.
/// This is also the recipe used by file/array Index construction; no I/O occurs.
pub fn resolve_options(
    options: &crate::IndexOptions,
    base: std::path::PathBuf,
    format: VectorFormat,
    dimension: usize,
    count: usize,
    content_id: Option<String>,
) -> crate::Result<ResolvedRunConfig> {
    options.resolve(base, format, dimension, count, content_id)
}

/// Validate a caller's query encoding against an already-resolved recipe.
pub fn validate_query_format(_config: &ResolvedRunConfig) -> crate::Result<()> {
    // Every supported query encoding is valid for either base storage type.
    Ok(())
}

/// Share supplier metadata validation with the CLI's offline preflight.
#[cfg(feature = "experimental-graph")]
pub fn configure_supplier(
    spec: &crate::supplier::PartitionSpec,
    options: &mut crate::IndexOptions,
) -> crate::Result<()> {
    spec.validate()?;
    crate::index::supplier::configure_options(spec, options)
}
