/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Header-only resource preflight. Metadata resolution, payload arithmetic,
//! budget enforcement and rendering have separate owners. These bounds exclude
//! unestimated scratch, allocator overhead and OS page cache; they are not RSS.
mod budget;
mod estimate;
mod metadata;
mod queries;
mod report;
#[cfg(test)]
mod sidecars;
#[cfg(all(test, feature = "cli"))]
mod tests;

pub(crate) use budget::{check_index_budget, check_selected_budget};
#[cfg(any(feature = "cli", feature = "benchmark", test))]
pub use budget::{check_memory_budget, enforce_budget};
#[cfg(feature = "cli")]
pub use report::print_report;

use crate::config::resources::BYTES_PER_GIB;
use crate::{config::ResolvedRunConfig, Result};
use estimate::{calculate_memory, checked_mul_bytes};
use metadata::resolve_memory_inputs;
#[cfg(any(feature = "cli", feature = "benchmark"))]
use metadata::resolve_vector_count;

/// Builds a resident-memory preflight report without loading vector or graph payloads.
///
/// The function validates available headers against declarations and falls back
/// to declared metadata for missing files. It estimates the
/// known resident-memory lower bound for the configured run.
///
/// The returned JSON object is intended for CLI reporting and run-log capture.
/// It is returned even when over budget; [`budget::enforce_budget`] performs that check.
///
/// # Errors
///
/// Returns an error if base scale is unknown, existing files cannot be read,
/// declared counts/dimensions conflict, graph metadata is invalid, or the estimate
/// overflows. Budget excess is recorded in the report.
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn preflight_report(config: &ResolvedRunConfig) -> Result<serde_json::Value> {
    let (base_count, base_source) = resolve_vector_count(
        &config.base,
        config.base_format,
        config.dimension,
        config.num_points,
        true,
    )?;
    let num_points = base_count.ok_or_else(|| format!(
        "{}: offline preflight needs declared num_points or a readable base file; max_points is only a loading limit",
        config.base.display()
    ))?.min(config.max_points);
    let input = resolve_memory_inputs(config, num_points, true, base_source)?;
    calculate_memory(config, input)
}

/// Estimates known resident-memory requirements for one resolved run.
///
/// This function accounts only for components whose resident size can be
/// determined from configuration and file/graph metadata without loading their
/// payloads. The result is therefore a lower bound rather than a prediction of
/// total process memory.
///
/// The estimate currently includes:
///
/// - resident base-vector storage,
/// - graph-slab storage,
/// - node-reader metadata,
/// - quantized admission and additional rerank buffers.
///
/// Temporary graph-construction state, sidecar build scratch,
/// allocator overhead, and OS page cache are intentionally excluded.
///
/// # Errors
///
/// Returns an error if the point count is invalid, graph metadata cannot be
/// resolved, arithmetic overflows, or the memory arithmetic overflows. Budget checking uses [`budget::enforce_budget`]
/// after the complete report is available.
pub fn estimate_resident_memory(
    config: &ResolvedRunConfig,
    num_points: usize,
) -> Result<serde_json::Value> {
    let base_source = if config.base_content_id.is_some() {
        "array"
    } else {
        "file"
    };
    let input = resolve_memory_inputs(config, num_points, false, base_source)?;
    calculate_memory(config, input)
}

pub(crate) fn graph_bytes(config: &ResolvedRunConfig, points: usize) -> Result<u64> {
    let input = resolve_memory_inputs(config, points, true, "configuration")?;
    checked_mul_bytes(points as u64, input.stride_bytes)
}
