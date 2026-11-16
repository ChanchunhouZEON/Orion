/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Enforce known payload bounds while retaining the full report on failure.
use super::{
    estimate::calculate_memory,
    estimate_resident_memory,
    metadata::{inputs_with_plan, resolve_memory_inputs, MemoryInputs},
    report::format_report,
};
use crate::{config::ResolvedRunConfig, graph::cache::CachePlan, Error, Result};
use std::io::Read;

/// Reject a known over-budget allocation before the base is loaded. Passing
/// this check is not a promise that builder scratch and sidecars will also fit.
pub(crate) fn check_index_budget(
    config: &ResolvedRunConfig,
    points: usize,
    snapshot_graph: Option<&std::path::Path>,
) -> Result<()> {
    let Some(path) = snapshot_graph else {
        if config.base_content_id.is_none() {
            // The streaming base decoder has not computed its content identity
            // yet, so a cache miss here is not authoritative. Estimate allocation
            // now; load() verifies the selected cache/source after hashing.
            let input = resolve_memory_inputs(config, points, true, "file header")?;
            return enforce_budget(&calculate_memory(config, input)?);
        }
        return check_memory_budget(config, points);
    };
    // A restored snapshot is a cache hit regardless of installed builders and
    // the availability of the original working directory or STAG input.
    let mut header = [0; 16];
    let mut file = std::fs::File::open(path)?;
    file.read_exact(&mut header)?;
    let nodes = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    let stride_bytes = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64 * 4;
    if nodes != points
        || (nodes as u64)
            .checked_mul(stride_bytes)
            .and_then(|v| v.checked_add(16))
            != Some(file.metadata()?.len())
    {
        return Err(Error::cache("snapshot graph shape or byte length mismatch"));
    }
    let input = MemoryInputs {
        num_points: points,
        base_source: "input",
        query_count: Some(0),
        query_source: "not used",
        stride_bytes,
        graph_source: "snapshot header",
        staged_source: None,
        cache_hit: true,
        cache_key: path.to_owned(),
    };
    enforce_budget(&calculate_memory(config, input)?)
}

pub(crate) fn check_selected_budget(
    config: &ResolvedRunConfig,
    points: usize,
    cache: &CachePlan,
) -> Result<()> {
    let input = inputs_with_plan(config, points, false, "input", cache)?;
    enforce_budget(&calculate_memory(config, input)?)
}

pub fn check_memory_budget(config: &ResolvedRunConfig, num_points: usize) -> Result<()> {
    let report = estimate_resident_memory(config, num_points)?;
    log::info!("Resource preflight:\n{}", format_report(&report));
    enforce_budget(&report)
}

pub fn enforce_budget(report: &serde_json::Value) -> Result<()> {
    if report["budget_exceeded"].as_bool() == Some(true) {
        return Err(Error::new(
            crate::ErrorCode::ResourceExhausted,
            format!(
                "memory budget exceeded (known lower bound):\n{}",
                format_report(report)
            ),
        )
        .at("preflight")
        .with_detail(
            "required_bytes",
            report["required_lower_bound_bytes"].clone(),
        )
        .with_detail("budget_gib", report["memory_budget_gib"].clone())
        .with_detail("budget_bytes", report["placement_limit_bytes"].clone())
        .with_detail("components", report["components"].clone()));
    }
    Ok(())
}
