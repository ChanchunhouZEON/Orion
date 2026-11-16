/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::BYTES_PER_GIB;
#[cfg(feature = "cli")]
use crate::{Error, Result};
/// Choose the presentation without changing the machine-readable report schema.
/// Auto keeps existing redirected JSON consumers working while terminal users get
/// a table. An explicit format also supports notebooks and captured terminal logs.
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn print_report(
    report: &serde_json::Value,
    format: crate::config::PreflightFormat,
) -> Result<()> {
    use crate::config::PreflightFormat;
    use std::io::IsTerminal;
    let table = match format {
        PreflightFormat::Auto => std::io::stdout().is_terminal(),
        PreflightFormat::Table => true,
        PreflightFormat::Json => false,
    };
    if table {
        println!("{}", format_report(report));
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(report).map_err(Error::from)?
        );
    }
    Ok(())
}

fn memory_size(bytes: u64) -> String {
    // Small query buffers should not disappear as rounded `0.000 GiB` values.
    for (unit, scale) in [
        ("TiB", 1u64 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    ] {
        if bytes >= scale {
            return format!("{:.3} {unit}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}

pub fn format_report(report: &serde_json::Value) -> String {
    use comfy_table::{presets::UTF8_FULL, Cell, CellAlignment, ContentArrangement, Table};
    let value = |key: &str| {
        report[key]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| report[key].to_string())
    };
    let mut text = format!(
        "Memory preflight — {}\nPoints: {} | Dimension: {} | Storage: {}\n\n",
        value("dataset"),
        value("num_points"),
        value("dimension"),
        value("vector_storage")
    );
    text.push_str(&format!(
        "Graph: {} → {} | {}\n{}\nCache: {}\n\n",
        value("graph_source_policy"),
        value("graph_source"),
        value("graph_action"),
        value("graph_selection_reason"),
        value("cache_path")
    ));
    if let Some(dispatch) = report["search_dispatch"].as_str() {
        text.push_str(&format!("Search dispatch: {dispatch}\n\n"));
    }
    if let Some(phase) = report["planned_phase"].as_str() {
        text.push_str(&format!("Planned phase: {phase}\n\n"));
    }
    let mut table = Table::new();
    let width = table.width().unwrap_or(112).min(112);
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(width)
        .set_header([
            "Component / phase",
            "Memory",
            "Source",
            "Calculation / assumptions",
        ]);
    if let Some(parts) = report["components"].as_array() {
        for part in parts {
            let amount = part["bytes"]
                .as_u64()
                .map(memory_size)
                .unwrap_or_else(|| "unknown".into());
            table.add_row(vec![
                Cell::new(format!(
                    "{}\n{}",
                    part["name"].as_str().unwrap_or("?"),
                    part["phase"].as_str().unwrap_or("?")
                )),
                Cell::new(amount).set_alignment(CellAlignment::Right),
                Cell::new(part["source"].as_str().unwrap_or("?")),
                Cell::new(part["basis"].as_str().unwrap_or("?")),
            ]);
        }
    }
    text.push_str(&table.to_string());
    if report["storage_options"]["objects"]
        .as_object()
        .is_some_and(|o| !o.is_empty())
    {
        let mut placements = Table::new();
        placements.load_preset(UTF8_FULL).set_header([
            "Object",
            "Requested",
            "Selected",
            "Payload",
        ]);
        if let Some(objects) = report["placement_objects"].as_array() {
            for object in objects {
                placements.add_row([
                    object["object"].as_str().unwrap_or("?"),
                    object["requested"].as_str().unwrap_or("?"),
                    object["placement"].as_str().unwrap_or("?"),
                    &memory_size(object["bytes"].as_u64().unwrap_or(0)),
                ]);
            }
        }
        text.push_str(&format!("\n\nStorage placement\n{placements}"));
        if let Some(inactive) = report["inactive_placement_overrides"]
            .as_array()
            .filter(|v| !v.is_empty())
        {
            let names = inactive
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            text.push_str(&format!(
                "\nOverrides outside this payload plan: {names} (no extra payload is allocated)\n"
            ));
        }
    }

    let known = report["accounted_lower_bound_bytes"].as_u64().unwrap_or(0) as f64 / BYTES_PER_GIB;
    text.push_str(&format!(
        "\n\nAccounted search lower bound: {known:.3} GiB\n"
    ));
    if let Some(mapped) = report["mapped_bytes"].as_u64().filter(|&n| n != 0) {
        text.push_str(&format!(
            "Mapped payload: {:.3} GiB (private backing files; pageable, not zero RSS)\n",
            mapped as f64 / BYTES_PER_GIB
        ));
        text.push_str(
            "Mapped payload limit: not configured; backing free space and quota are not checked.\n",
        );
    }
    let required =
        report["required_lower_bound_bytes"].as_u64().unwrap_or(0) as f64 / BYTES_PER_GIB;
    let build = report["build_lower_bound_bytes"].as_u64().unwrap_or(0) as f64 / BYTES_PER_GIB;
    text.push_str(&format!(
        "Accounted build/import lower bound: {build:.3} GiB\n"
    ));
    let requested = report["memory_budget_gib"].as_f64();
    let effective = report["placement_limit_bytes"]
        .as_u64()
        .map(|bytes| bytes as f64 / BYTES_PER_GIB)
        .or(requested);
    if let Some(budget) = effective {
        let status = if report["budget_exceeded"].as_bool() == Some(true) {
            "EXCEEDED"
        } else {
            "known components within budget"
        };
        text.push_str(&format!(
            "Budget: {budget:.3} GiB | excess: {:.3} GiB | {status}\n",
            (required - budget).max(0.0)
        ));
        if let Some(requested) = requested.filter(|r| (*r - budget).abs() > 1.0 / BYTES_PER_GIB) {
            text.push_str(&format!(
                "Requested budget: {requested:.3} GiB; effective limit shown above.\n"
            ));
        }
    } else {
        text.push_str("Budget: not specified (estimate only)\n");
    }
    if report["planned_phase"].as_str() == Some("graph-only") {
        text.push_str("Graph-only report: search sidecars are excluded; run search-ready preflight before calibration/search.\n");
    }
    text.push_str("Unknown components are not zero. Build and search are separate phases; this is not a peak-RSS guarantee.");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn excess_uses_effective_budget_instead_of_larger_requested_limit() {
        let report = serde_json::json!({
            "required_lower_bound_bytes": 3u64 << 28,
            "memory_budget_gib": 1.0,
            "placement_limit_bytes": 1u64 << 29,
            "budget_exceeded": true
        });
        let text = format_report(&report);
        assert!(text.contains("Budget: 0.500 GiB | excess: 0.250 GiB | EXCEEDED"));
        assert!(text.contains("Requested budget: 1.000 GiB"));
    }
}
