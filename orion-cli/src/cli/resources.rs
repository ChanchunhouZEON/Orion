/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resource preflight for estimating known resident-memory requirements.
//!
//! This module inspects vector-file metadata, graph-cache headers, and staged
//! graph headers without loading the corresponding payloads. It computes a
//! lower bound for memory that is known to remain resident during index use.
//!
//! The estimate currently includes:
//!
//! - resident base-vector storage,
//! - the graph slab,
//! - node-reader metadata,
//! - the `L2U8` admission sidecar when one is required.
//!
//! It intentionally excludes temporary construction state, most cascade
//! sidecars, allocator/runtime overhead, query scratch, and OS page cache.
//! Therefore, the reported value is a lower bound and must not be interpreted
//! as predicted peak RSS.
use super::{
    cache::CachePlan,
    config::{GraphSource, ResolvedRunConfig, VectorStorageKind},
    data::inspect_vector_file,
};
use std::{io::Read, path::PathBuf};

const BYTES_PER_GIB: f64 = 1_073_741_824.0;

// Number of `u32` words stored before adjacency entries in each graph record.
const GRAPH_HEADER_WORDS: u64 = 4;

/// Resident graph records are padded to this cache-line size.
const CACHE_LINE_BYTES: u64 = 64;

/// Extra tail allocation reserved for SIMD reads past the final vector.
const SIMD_PAD_BYTES: u64 = 64;

/// Builds a resident-memory preflight report without loading vector or graph payloads.
///
/// The function inspects only file headers and graph metadata, validates the
/// effective dataset shape against [`ResolvedRunConfig`], and estimates the
/// known resident-memory lower bound for the configured run.
///
/// The returned JSON object is intended for CLI reporting and run-log capture.
/// It is returned even when over budget; [`enforce_budget`] performs that check.
///
/// # Errors
///
/// Returns an error if vector metadata cannot be read, dimensions are
/// inconsistent, graph metadata is invalid, or the resolved memory estimate
/// overflows. Budget excess is recorded in the report.
pub fn preflight_report(config: &ResolvedRunConfig) -> Result<serde_json::Value, String> {
    let base = inspect_vector_file(&config.base, config.base_format)?;
    if base.dimension != config.dimension {
        return Err(format!(
            "{}: base dimension {} != configured dimension {}",
            config.base.display(),
            base.dimension,
            config.dimension
        ));
    }
    if !config.prepare_only {
        if let Some(path) = config
            .query
            .as_deref()
            .filter(|p| *p != std::path::Path::new("-"))
        {
            let metadata =
                std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
            if metadata.is_file() {
                let query = inspect_vector_file(path, config.query_format)?;
                if query.dimension != base.dimension {
                    return Err(format!(
                        "{}: query dimension {} != base dimension {}",
                        path.display(),
                        query.dimension,
                        base.dimension
                    ));
                }
            }
        }
    }

    estimate_resident_memory(config, base.count.min(config.max_points))
}

/// Computes the cache-line-aligned node stride for [`PhasedGraph`](orion::model::PhasedGraph).
///
/// `max_degree` is the combined capacity of the local and remote adjacency
/// regions; `max_extra` is the capacity of the extra-candidate region.
fn graph_stride_bytes(max_degree: u64, max_extra: u64) -> u64 {
    // Each node stores header words and both adjacency regions, rounded to a cache line.
    let record_bytes = (GRAPH_HEADER_WORDS + max_degree + max_extra) * 4;
    (record_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES
}

/// Resolves the resident per-node stride used by [`PhasedGraph`](orion::model::PhasedGraph)
/// for memory preflight.
///
/// The stride is obtained from the most authoritative representation available:
///
/// 1. an existing graph cache, whose header already stores the padded stride;
/// 2. a ParlayANN/STAG export, whose recorded capacities are converted into the
///    corresponding [`PhasedGraph`](orion::model::PhasedGraph) layout;
/// 3. the resolved graph-construction settings when no materialized graph exists.
///
/// The primary graph capacity is shared by the local and remote neighbor regions,
/// while `max_extra` determines the separately reserved extra-candidate region.
/// Runtime occupancy counters such as the local or extra count do not change the
/// fixed per-node allocation.
///
/// Returns `(stride_bytes, staged_source)`, where `staged_source` is present only
/// when a staged ParlayANN export had to be inspected.
///
/// # Errors
///
/// Returns an error if required graph metadata cannot be read, a staged export
/// cannot be resolved, or its capacities are incompatible with the resolved
/// configuration.
fn resolve_graph_layout(
    config: &ResolvedRunConfig,
    cache: &CachePlan,
    num_points: usize,
) -> Result<(u64, Option<PathBuf>), String> {
    if cache.is_hit {
        let mut header = [0; 16];
        std::fs::File::open(cache.metadata_path.with_extension("pgraph"))
            .and_then(|mut file| file.read_exact(&mut header))
            .map_err(|e| e.to_string())?;

        // The cached graph header stores its already-padded stride in u32 words.
        let stride_words = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
        return Ok((stride_words * 4, None));
    }

    if config.graph_source == GraphSource::Parlayann {
        let staged_path = CachePlan::resolve_staged_export(config, num_points)?;
        let mut header = [0; 24];
        std::fs::File::open(&staged_path)
            .and_then(|mut file| file.read_exact(&mut header))
            .map_err(|e| e.to_string())?;

        // STAG stores capacities, so reconstruct the resident graph stride from them.
        let max_degree = u32::from_le_bytes(header[12..16].try_into().unwrap()) as u64;
        let max_extra = u32::from_le_bytes(header[16..20].try_into().unwrap()) as u64;

        if max_degree != config.graph_degree as u64
            || max_degree > num_points as u64
            || max_extra > num_points as u64
        {
            return Err("staged capacities/config degree mismatch".into());
        }
        return Ok((graph_stride_bytes(max_degree, max_extra), Some(staged_path)));
    }

    Ok((
        graph_stride_bytes(config.graph_degree as u64, config.max_extra as u64),
        None,
    ))
}

fn checked_mul_bytes(count: u64, bytes_per_item: u64) -> Result<u64, String> {
    count
        .checked_mul(bytes_per_item)
        .ok_or_else(|| "memory estimate overflow".into())
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
/// - the `L2U8` admission representation when it is separately allocated.
///
/// Temporary graph-construction state, most other cascade sidecars, query data,
/// allocator overhead, and OS page cache are intentionally excluded.
///
/// # Errors
///
/// Returns an error if the point count is invalid, graph metadata cannot be
/// resolved, arithmetic overflows, or the memory arithmetic overflows. Budget checking uses [`enforce_budget`]
/// after the complete report is available.
fn estimate_resident_memory(
    config: &ResolvedRunConfig,
    num_points: usize,
) -> Result<serde_json::Value, String> {
    if num_points == 0
        || num_points > u32::MAX as usize
        || (!config.prepare_only && config.sweep.k > num_points)
    {
        return Err(format!("invalid loaded point count {num_points} or k={}: require 1..={} points and k <= loaded count for search (base {}, max-points {})", config.sweep.k, u32::MAX, config.base.display(), config.max_points));
    }

    let cache = CachePlan::resolve(config, num_points)?;
    let (stride_bytes, staged_source) = resolve_graph_layout(config, &cache, num_points)?;

    // Resident storage is selected explicitly. A native-byte admission stage
    // borrows this buffer; it does not allocate a second quantized dataset.
    let coordinate_count = checked_mul_bytes(num_points as u64, config.dimension as u64)?;
    let storage = config.search_plan().storage();
    let coordinate_bytes = storage.element_bytes as u64;
    let base_payload_bytes = checked_mul_bytes(coordinate_count, coordinate_bytes)?;
    let base_bytes = base_payload_bytes
        .checked_add(SIMD_PAD_BYTES)
        .ok_or("memory estimate overflow")?;
    let graph_bytes = checked_mul_bytes(num_points as u64, stride_bytes)?;
    let node_reader_bytes =
        checked_mul_bytes(num_points as u64, std::mem::size_of::<usize>() as u64)?;

    // Only the L2U8 representation is accounted for here. Other cascade buffers
    // remain explicitly excluded, rather than presented as a complete estimate.
    let admission_bytes = if storage.has_l2_u8_sidecar && !config.prepare_only {
        let aligned_dimension = (config.dimension as u64 + 31) / 32 * 32;
        checked_mul_bytes(num_points as u64, aligned_dimension)?
    } else {
        0
    };

    let resident_lower_bound = base_bytes
        .checked_add(admission_bytes)
        .and_then(|bytes| bytes.checked_add(graph_bytes))
        .and_then(|bytes| bytes.checked_add(node_reader_bytes))
        .ok_or("memory estimate overflow")?;

    // Report known working buffers separately from resident index storage.
    // Unknown lifetimes/implementations remain explicit rather than counted as zero.
    let query_count = if config.prepare_only {
        Some(0)
    } else {
        config
            .query
            .as_deref()
            .filter(|p| *p != std::path::Path::new("-"))
            .map(|path| -> Result<Option<usize>, String> {
                let meta =
                    std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
                if meta.is_file() {
                    Ok(Some(inspect_vector_file(path, config.query_format)?.count))
                } else {
                    Ok(None)
                }
            })
            .transpose()?
            .flatten()
    };
    let buffered = if config.prepare_only {
        0
    } else if config.replay_queries {
        query_count
            .ok_or("sweep requires a regular, replayable query file; use orion for a stream")?
    } else {
        query_count
            .unwrap_or(
                config
                    .query_batch_size
                    .max(config.sweep.calibration_samples),
            )
            .min(
                config
                    .query_batch_size
                    .max(config.sweep.calibration_samples),
            )
    };
    let query_bytes = checked_mul_bytes(
        checked_mul_bytes(buffered as u64, config.dimension as u64)?,
        8,
    )?;
    let result_bytes = checked_mul_bytes(
        checked_mul_bytes(buffered as u64, config.sweep.k as u64)?,
        4,
    )?;
    let gt_bytes = if config.groundtruth.is_some() && !config.prepare_only {
        checked_mul_bytes(
            if config.replay_queries {
                query_count.unwrap_or(0) as u64
            } else {
                1
            },
            config.sweep.k as u64 * 4,
        )?
    } else {
        0
    };
    let accounted = resident_lower_bound
        .checked_add(query_bytes)
        .and_then(|v| v.checked_add(result_bytes))
        .and_then(|v| v.checked_add(gt_bytes))
        .ok_or("memory estimate overflow: query/result/GT buffers")?;
    let exceeded = config
        .memory_budget_gib
        .is_some_and(|budget| accounted as f64 > budget * BYTES_PER_GIB);
    let cascade = config.search_plan().cascade();
    let other_sidecars = cascade.prefilter != crate::cascade::PrefilterChoice::None
        || !matches!(
            cascade.admission,
            crate::cascade::AdmissionChoice::NativeL2U8
                | crate::cascade::AdmissionChoice::L2U8
                | crate::cascade::AdmissionChoice::AdsF32
        )
        || cascade.rerank == crate::cascade::RerankChoice::U16;
    let components = serde_json::json!([
        {"name":"base vectors", "bytes":base_bytes, "basis":format!("{num_points} × {} × {} bytes + SIMD tail", config.dimension, storage.element_bytes), "phase":"resident"},
        {"name":"graph slab", "bytes":graph_bytes, "basis":format!("{num_points} × {stride_bytes} bytes/node"), "phase":"resident"},
        {"name":"node reader counters", "bytes":node_reader_bytes, "basis":"one usize per node", "phase":"resident"},
        {"name":"L2U8 admission sidecar", "bytes":admission_bytes, "basis":if storage.has_l2_u8_sidecar {"aligned byte vectors"} else {"not allocated for this phase/recipe; native admission shares base"}, "phase":"resident"},
        {"name":"query buffers", "bytes":query_bytes, "basis":format!("up to {buffered} rows × dimension × f32 × 2; decoding plus search arrays"), "phase":"search"},
        {"name":"result IDs", "bytes":result_bytes, "basis":"buffered queries × k × u32", "phase":"search"},
        {"name":"ground truth IDs", "bytes":gt_bytes, "basis":"replay keeps all top-k rows; streaming keeps one row; absent GT uses none", "phase":"evaluation"},
        {"name":"other cascade sidecars", "bytes":if other_sidecars {None} else {Some(0u64)}, "basis":"selected prefilter/admission/rerank; not yet estimated when allocated", "phase":"resident"},
        {"name":"worker scratch", "bytes":null, "basis":format!("{} workers; beam schedule {:?}; visited sets and candidate buffers grow dynamically", config.sweep.threads, config.sweep.search_list_sizes), "phase":"search"},
        {"name":"graph construction temporary memory", "bytes":null, "basis":if cache.is_hit {"cache load; decoder and allocator overhead not estimated"} else if config.graph_source == GraphSource::Parlayann {"streamed STAG import; no in-process builder"} else {"Rust builder retains working dataset and graph partitions; peak not estimated"}, "phase":"build/import"},
        {"name":"allocator/runtime/OS overhead", "bytes":null, "basis":"not included in payload bounds", "phase":"all"}
    ]);

    // Retain output field names for existing run-log consumers.
    Ok(serde_json::json!({
        "dataset": config.dataset,
        "components": components,
        "query_count": query_count,
        "query_buffered_rows": buffered,
        "accounted_lower_bound_bytes": accounted,
        "budget_exceeded": exceeded,
        "num_points": num_points,
        "dimension": config.dimension,
        "base_format": config.base_format,
        "vector_storage": storage.kind,
        "base_storage_bytes": base_bytes,
        "base_f32_bytes": if storage.kind == VectorStorageKind::F32 { base_bytes } else { 0 },
        "base_u8_bytes": if storage.kind == VectorStorageKind::U8 { base_bytes } else { 0 },
        "graph_slab_bytes": graph_bytes,
        "l2_u8_admission_bytes": admission_bytes,
        "node_reader_bytes": node_reader_bytes,
        "resident_lower_bound_bytes": resident_lower_bound,
        "resident_lower_bound_gib": resident_lower_bound as f64 / BYTES_PER_GIB,
        "cache_hit": cache.is_hit,
        "cache_path": cache.metadata_path,
        "staged_source": staged_source,
        "memory_budget_gib": config.memory_budget_gib,
        "excludes": [
            "sidecars other than L2U8 admission, and sidecar construction overhead",
            "query/result/GT container and allocator overhead beyond counted payloads",
            "query scratch",
            "allocator/runtime overhead",
            "in-process graph construction",
            "OS page cache"
        ],
        "note": "Lower bound only. Budget check is not a peak-RSS guarantee. \
                 Measure a smaller run before a full build. \
                 A PA cache miss requires an existing staged export."
    }))
}

/// Reject a known over-budget allocation before the base is loaded. Passing
/// this check is not a promise that builder scratch and sidecars will also fit.
pub fn check_memory_budget(config: &ResolvedRunConfig, num_points: usize) -> Result<(), String> {
    let report = estimate_resident_memory(config, num_points)?;
    log::info!("Resource preflight:\n{}", format_report(&report));
    enforce_budget(&report)
}

pub fn enforce_budget(report: &serde_json::Value) -> Result<(), String> {
    if report["budget_exceeded"].as_bool() == Some(true) {
        return Err(format!(
            "memory budget exceeded (known lower bound):\n{}",
            format_report(report)
        ));
    }
    Ok(())
}

pub fn format_report(report: &serde_json::Value) -> String {
    let mut text = format!(
        "Dataset: {}; points: {}; dimension: {}; storage: {}\n",
        report["dataset"], report["num_points"], report["dimension"], report["vector_storage"]
    );
    if let Some(parts) = report["components"].as_array() {
        for part in parts {
            let amount = part["bytes"]
                .as_u64()
                .map(|v| format!("{:.3} GiB ({v} bytes)", v as f64 / BYTES_PER_GIB))
                .unwrap_or_else(|| "unknown / not estimated".into());
            text.push_str(&format!(
                "  {}: {amount} [{}] — {}\n",
                part["name"].as_str().unwrap_or("?"),
                part["phase"].as_str().unwrap_or("?"),
                part["basis"].as_str().unwrap_or("?")
            ));
        }
    }
    let known = report["accounted_lower_bound_bytes"].as_u64().unwrap_or(0) as f64 / BYTES_PER_GIB;
    text.push_str(&format!("Accounted search lower bound: {known:.3} GiB\n"));
    if let Some(budget) = report["memory_budget_gib"].as_f64() {
        text.push_str(&format!(
            "Budget: {budget:.3} GiB; excess: {:.3} GiB\n",
            (known - budget).max(0.0)
        ));
    }
    text.push_str("Unknown components are not zero. Build and search are separate phases; this is not a peak-RSS guarantee.");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn native_billion_estimate_counts_one_byte_base_and_no_admission_copy() {
        let directory =
            std::env::temp_dir().join(format!("orion-native-estimate-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let staged = directory.join("graph.staged");
        let header: Vec<u8> = [0x53544147u32, 3, 1_000_000_000, 64, 16, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        std::fs::write(&staged, header).unwrap();
        let args = super::super::config::Args::try_parse_from([
            "orion",
            "sift1b",
            "--query-format",
            "bvecs",
            "--query",
            "-",
            "--staged-file",
            staged.to_str().unwrap(),
            "--cache-dir",
            directory.to_str().unwrap(),
        ])
        .unwrap();
        let config = args.resolve().unwrap();
        let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
        assert_eq!(report["base_f32_bytes"], 0);
        assert_eq!(report["base_u8_bytes"], 128_000_000_064u64);
        assert_eq!(report["l2_u8_admission_bytes"], 0);
        assert_eq!(report["resident_lower_bound_bytes"], 520_000_000_064u64);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn billion_lower_bound_and_budget_guard() {
        let args = super::super::config::Args::try_parse_from([
            "orion",
            "--base",
            "missing-large.bvecs",
            "--query",
            "-",
            "--groundtruth",
            "missing.ivecs",
            "--dimension",
            "128",
            "--metric",
            "l2",
            "--graph-source",
            "rust",
            "--cache-dir",
            "/tmp/orion-resource-unit-test-unused",
        ])
        .unwrap();
        let mut config = args.resolve().unwrap();
        let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
        assert_eq!(report["base_f32_bytes"], 512_000_000_064u64);
        assert_eq!(report["graph_slab_bytes"], 384_000_000_000u64);
        assert_eq!(report["l2_u8_admission_bytes"], 128_000_000_000u64);
        config.memory_budget_gib = Some(128.);
        let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
        let error = enforce_budget(&report).unwrap_err();
        for component in [
            "base vectors",
            "graph slab",
            "node reader counters",
            "query buffers",
            "worker scratch",
            "Budget:",
            "excess:",
            "unknown",
        ] {
            assert!(error.contains(component), "missing {component}: {error}");
        }
        assert!(report["components"].as_array().unwrap().len() >= 10);
    }
}
