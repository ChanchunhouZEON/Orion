/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resource preflight for estimating known resident-memory requirements.
//!
//! Offline planning uses declared counts and graph settings when files are absent.
//! Available vector/cache/export headers are validated without loading payloads.
//! Runtime budget checks still require the inputs needed for execution. It computes a
//! lower bound for memory that is known to remain resident during index use.
//!
//! The estimate currently includes:
//!
//! - resident base-vector storage,
//! - the graph slab,
//! - node-reader metadata,
//! - all requested quantized/projected sidecars, counting shared storage once.
//!
//! It intentionally excludes temporary construction state, allocator/runtime
//! overhead, query scratch, and OS page cache.
//! Therefore, the reported value is a lower bound and must not be interpreted
//! as predicted peak RSS.
use crate::{
    config::{GraphSource, ResolvedRunConfig, VectorStorageKind},
    data::inspect_vector_file,
    graph::cache::CachePlan,
};
use crate::{Error, Result};
use std::{io::Read, path::PathBuf};

mod report;
#[cfg(test)]
mod sidecars;
use report::format_report;
#[cfg(feature = "cli")]
pub use report::print_report;

use crate::config::resources::BYTES_PER_GIB;

// Number of `u32` words stored before adjacency entries in each graph record.
const GRAPH_HEADER_WORDS: u64 = 4;

/// Resident graph records are padded to this cache-line size.
const CACHE_LINE_BYTES: u64 = 64;

/// Resolved counts and graph layout consumed by the pure budget calculator.
/// Report/budget entry points follow; metadata resolution and arithmetic come last.
struct MemoryInputs {
    num_points: usize,
    base_source: &'static str,
    query_count: Option<usize>,
    query_source: &'static str,
    stride_bytes: u64,
    graph_source: &'static str,
    staged_source: Option<PathBuf>,
    cache_hit: bool,
    cache_key: PathBuf,
}

/// Builds a resident-memory preflight report without loading vector or graph payloads.
///
/// The function validates available headers against declarations and falls back
/// to preset metadata for missing files. It estimates the
/// known resident-memory lower bound for the configured run.
///
/// The returned JSON object is intended for CLI reporting and run-log capture.
/// It is returned even when over budget; [`enforce_budget`] performs that check.
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
        "{}: offline preflight needs preset num_points or a readable base file; max_points is only a loading limit",
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
/// resolved, arithmetic overflows, or the memory arithmetic overflows. Budget checking uses [`enforce_budget`]
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

/// Reject a known over-budget allocation before the base is loaded. Passing
/// this check is not a promise that builder scratch and sidecars will also fit.
pub(crate) fn check_index_budget(
    config: &ResolvedRunConfig,
    points: usize,
    snapshot_graph: Option<&std::path::Path>,
) -> Result<()> {
    let Some(path) = snapshot_graph else {
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
        .with_detail("components", report["components"].clone()));
    }
    Ok(())
}

/// Missing files are allowed only during offline planning. Existing invalid files
/// always fail, so a declaration cannot hide corruption or a mismatched dataset.
fn resolve_vector_count(
    path: &std::path::Path,
    format: crate::data::VectorFormat,
    dimension: usize,
    declared: Option<usize>,
    offline: bool,
) -> Result<(Option<usize>, &'static str)> {
    if path == std::path::Path::new("-") {
        return Ok((None, "stream"));
    }
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => {
            let actual = inspect_vector_file(path, format)?;
            if actual.dimension != dimension {
                return Err(Error::from(format!(
                    "{}: file dimension {} != configured dimension {dimension}",
                    path.display(),
                    actual.dimension
                )));
            }
            if declared.is_some_and(|count| count != actual.count) {
                return Err(Error::from(format!(
                    "{}: file count {} != declared count {}; update the preset or explicitly override the input path",
                    path.display(),
                    actual.count,
                    declared.unwrap()
                )));
            }
            Ok((Some(actual.count), "file"))
        }
        Ok(_) => Ok((None, "stream")),
        Err(error) if offline && error.kind() == std::io::ErrorKind::NotFound => Ok((
            declared,
            if declared.is_some() {
                "configuration"
            } else {
                "unknown"
            },
        )),
        Err(error) => Err(Error::from(format!("{}: {error}", path.display()))),
    }
}

/// Resolve all I/O before arithmetic. The execution path uses the same calculator
/// but does not permit missing query files or graph exports on a cache miss.
fn resolve_memory_inputs(
    config: &ResolvedRunConfig,
    num_points: usize,
    offline: bool,
    base_source: &'static str,
) -> Result<MemoryInputs> {
    let cache = if offline {
        CachePlan::resolve(config, num_points)?
    } else {
        CachePlan::inspect(config, num_points)?
    };
    inputs_with_plan(config, num_points, offline, base_source, &cache)
}

fn inputs_with_plan(
    config: &ResolvedRunConfig,
    num_points: usize,
    offline: bool,
    base_source: &'static str,
    cache: &CachePlan,
) -> Result<MemoryInputs> {
    let (stride_bytes, staged_source) = resolve_graph_layout(config, cache, num_points, offline)?;
    let graph_source = if cache.is_hit() {
        "cache header"
    } else if staged_source.is_some() {
        "staged header"
    } else {
        "configuration"
    };
    let (query_count, query_source) = if config.prepare_only {
        (Some(0), "not used")
    } else if let Some(path) = &config.query {
        resolve_vector_count(
            path,
            config.query_format,
            config.dimension,
            config.num_queries,
            offline,
        )?
    } else {
        (None, "unknown")
    };
    if !offline && config.replay_queries && query_count.is_none() {
        return Err(
            "sweep requires a regular, replayable query file; use orion for a stream".into(),
        );
    }
    Ok(MemoryInputs {
        num_points,
        base_source,
        query_count,
        query_source,
        stride_bytes,
        graph_source,
        staged_source,
        cache_hit: cache.is_hit(),
        cache_key: cache.metadata_path.clone(),
    })
}

/// Computes the cache-line-aligned node stride for internal graph slab.
///
/// `max_degree` is the combined capacity of the local and remote adjacency
/// regions; `max_extra` is the capacity of the extra-candidate region.
fn graph_stride_bytes(max_degree: u64, max_extra: u64) -> u64 {
    // Each node stores header words and both adjacency regions, rounded to a cache line.
    let record_bytes = (GRAPH_HEADER_WORDS + max_degree + max_extra) * 4;
    (record_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES
}

/// Resolves the resident per-node stride used by internal graph slab
/// for memory preflight.
///
/// The stride is obtained from the most authoritative representation available:
///
/// 1. an existing graph cache, whose header already stores the padded stride;
/// 2. a ParlayANN/STAG export, whose recorded capacities are converted into the
///    corresponding internal graph slab layout;
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
    offline: bool,
) -> Result<(u64, Option<PathBuf>)> {
    if cache.is_hit() {
        let mut header = [0; 16];
        std::fs::File::open(cache.graph_path()?)
            .and_then(|mut file| file.read_exact(&mut header))
            .map_err(Error::from)?;

        // The cached graph header stores its already-padded stride in u32 words.
        let stride_words = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
        return Ok((stride_words * 4, None));
    }

    // Only a genuinely absent export permits configuration-based planning.
    // Permission errors and existing malformed exports remain errors.
    let export_missing = config.graph_source != GraphSource::Parlayann
        || crate::graph::staged::available_staged_export(config)?.is_none();
    if config.graph_source == GraphSource::Parlayann
        && !(export_missing
            && (offline || (cfg!(feature = "parlay") && !config.staged_file_explicit)))
    {
        let staged_path = crate::graph::staged::resolve_staged_export(config, num_points)?;
        let mut header = [0; 24];
        std::fs::File::open(&staged_path)
            .and_then(|mut file| file.read_exact(&mut header))
            .map_err(Error::from)?;

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

fn checked_mul_bytes(count: u64, bytes_per_item: u64) -> Result<u64> {
    count
        .checked_mul(bytes_per_item)
        .ok_or_else(|| "memory estimate overflow".into())
}

/// Pure accounting shared by offline planning and strict runtime budget checks.
fn calculate_memory(config: &ResolvedRunConfig, input: MemoryInputs) -> Result<serde_json::Value> {
    let MemoryInputs {
        num_points,
        base_source,
        query_count,
        query_source,
        stride_bytes,
        graph_source,
        staged_source,
        cache_hit,
        cache_key,
    } = input;
    if num_points == 0
        || num_points > u32::MAX as usize
        || (!config.prepare_only && config.sweep.k > num_points)
    {
        return Err(Error::from(format!(
            "invalid loaded point count {num_points} or k={}: require 1..={} points and k <= loaded count for search (base {}, max-points {})",
            config.sweep.k,
            u32::MAX,
            config.base.display(),
            config.max_points
        )));
    }

    // Resident storage is selected explicitly. A native-byte admission stage
    // borrows this buffer; it does not allocate a second quantized dataset.
    let base_representation = config.base_representation();
    let storage = base_representation.kind();
    let layout = match storage {
        VectorStorageKind::F32 => orion_ann_vector::PaddedLayout::for_kernel::<
            orion_ann_vector::L2F32Distance,
        >(config.dimension)?,
        VectorStorageKind::U8 => orion_ann_vector::PaddedLayout::for_kernel::<
            orion_ann_vector::L2U8Distance,
        >(config.dimension)?,
    };
    let base_bytes = checked_mul_bytes(num_points as u64, layout.stride_bytes() as u64)?;
    let graph_bytes = checked_mul_bytes(num_points as u64, stride_bytes)?;
    // Graph storage is immutable during search; no per-node reader array is allocated.
    let node_reader_bytes = 0u64;

    let requirements = config
        .index_plan()
        .sidecar_requirements(config.prepare_only);
    let sidecars = crate::sidecars::requirements::estimate_sidecars(
        orion_ann_core::QuantizedDatasetShape {
            rows: num_points,
            dimension: config.dimension,
        },
        &requirements,
    )?;

    // Report known working buffers separately from resident index storage.
    // Unknown lifetimes/implementations remain explicit rather than counted as zero.
    let unknown_replay_buffers =
        config.replay_queries && query_count.is_none() && !config.prepare_only;
    let buffered = if config.prepare_only {
        0
    } else if config.replay_queries {
        query_count.unwrap_or(0)
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
    let mut query_bytes = checked_mul_bytes(
        checked_mul_bytes(buffered as u64, config.dimension as u64)?,
        8,
    )?;
    if storage == VectorStorageKind::U8 {
        let native_bytes = checked_mul_bytes(
            buffered.min(config.sweep.threads) as u64,
            layout.stride_bytes() as u64,
        )?;
        query_bytes = query_bytes
            .checked_add(native_bytes)
            .ok_or("query buffers overflow")?;
    }
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
    // Include known query buffers before selecting mappings: an owned sidecar
    // may fit by itself while leaving no room for the requested query batch.
    let fixed_bytes = base_bytes
        .checked_add(graph_bytes)
        .and_then(|v| v.checked_add(query_bytes))
        .and_then(|v| v.checked_add(result_bytes))
        .and_then(|v| v.checked_add(gt_bytes))
        .ok_or("memory estimate overflow: fixed buffers")?;
    // Offline reports describe the supplied budget, which may target another
    // machine. Live preparation separately probes current memory headroom.
    let placement = crate::placement::complete_plan(
        config,
        orion_ann_core::QuantizedDatasetShape {
            rows: num_points,
            dimension: config.dimension,
        },
        base_bytes,
        graph_bytes,
        fixed_bytes - base_bytes - graph_bytes,
        orion_ann_platform::resources::MemoryResources::default(),
    )?;
    let resident_lower_bound = placement.resident_bytes;
    let accounted = resident_lower_bound
        .checked_add(placement.reserve_bytes)
        .and_then(|v| v.checked_add(query_bytes))
        .and_then(|v| v.checked_add(result_bytes))
        .and_then(|v| v.checked_add(gt_bytes))
        .ok_or("memory estimate overflow: query/result/GT buffers")?;
    // ParlayANN borrows the final slab for both adjacency and extras. There is
    // no independent N-sized graph/extras allocation in the production path.
    // Sidecars are prepared later; candidate/statistics scratch is listed below.
    let working_bytes = 0u64;
    let owned_base = if placement.options.contains_key("base") {
        0
    } else {
        base_bytes
    };
    let owned_graph = if placement.options.contains_key("graph") {
        0
    } else {
        graph_bytes
    };
    let build_lower_bound = owned_base
        .checked_add(owned_graph)
        .and_then(|v| v.checked_add(node_reader_bytes))
        .and_then(|v| v.checked_add(working_bytes))
        .ok_or("memory estimate overflow: ParlayANN build")?;
    let required = accounted.max(build_lower_bound);
    let exceeded = placement
        .limit_bytes
        .is_some_and(|budget| required > budget);
    let mut components = serde_json::json!([
        {"name":"base vectors", "bytes":base_bytes, "basis":format!("{num_points} × {} bytes/row ({} logical coordinates × {} bytes, SIMD-padded)", layout.stride_bytes(), config.dimension, base_representation.element_bytes()), "phase": if owned_base == 0 { "mapped" } else { "resident" }},
        {"name":"graph slab", "bytes":graph_bytes, "basis":format!("{num_points} × {stride_bytes} bytes/node"), "phase": if owned_graph == 0 { "mapped" } else { "resident" }},
        {"name":"node reader counters", "bytes":node_reader_bytes, "basis":"no per-node counters; exclusive graph mutation", "phase":"resident"},
        {"name":"query buffers", "bytes":if unknown_replay_buffers { None } else { Some(query_bytes) }, "basis":if unknown_replay_buffers { "query count unknown; full replay payload cannot be estimated".to_string() } else { format!("up to {buffered} rows × dimension × f32 × 2; decoding/search arrays{}", if storage == VectorStorageKind::U8 { "; plus one SIMD-padded byte query per active worker" } else { "" }) }, "phase":"search"},
        {"name":"result IDs", "bytes":if unknown_replay_buffers { None } else { Some(result_bytes) }, "basis":"buffered queries × k × u32", "phase":"search"},
        {"name":"ground truth IDs", "bytes":if unknown_replay_buffers && config.groundtruth.is_some() { None } else { Some(gt_bytes) }, "basis":"replay keeps all top-k rows; streaming keeps one row; absent GT uses none", "phase":"evaluation"},
        {"name":"worker scratch", "bytes":null, "basis":format!("{} workers; beam schedule {:?}; visited sets and candidate buffers grow dynamically", config.sweep.threads, config.sweep.search_list_sizes), "phase":"search"},
        {"name":"ParlayANN working graph and extras", "bytes":working_bytes, "basis":"adjacency and extras share the final graph slab; no independent graph allocation", "phase":"build/import"},
        {"name":"graph construction temporary memory", "bytes":null, "basis":if cache_hit {"cache load; decoder and allocator overhead not estimated"} else if config.graph_source == GraphSource::Parlayann {
            if crate::graph::staged::available_staged_export(config)?.is_some() { "streamed STAG import; no in-process builder" }
            else { "ParlayANN borrows base and graph slab; construction statistics, insertion batches and worker candidate/partition scratch remain additional" }
        } else {"Rust builder retains working dataset and graph partitions; peak not estimated"}, "phase":"build/import"},
        {"name":"allocator/runtime/OS overhead", "bytes":null, "basis":"not included in payload bounds", "phase":"all"}
    ]);

    // Retain output field names for existing run-log consumers.
    for part in components.as_array_mut().unwrap() {
        part["source"] = serde_json::json!(match part["name"].as_str().unwrap() {
            "base vectors" | "node reader counters" => base_source,
            "graph slab" => graph_source,
            "ParlayANN working graph and extras" => "configuration",
            "query buffers" | "result IDs" | "ground truth IDs" => query_source,
            _ => "not estimated",
        });
    }
    // Dataset owners supply all payload components; stage labels never affect
    // allocation identity or accounting.
    for allocation in sidecars.allocations() {
        let memory = allocation.memory;
        components.as_array_mut().unwrap().push(serde_json::json!({
            "name": format!("{:?} sidecar", allocation.requirement.kind()),
            "bytes": memory.total_bytes(),
            "placement": if placement.options.contains_key(crate::placement::codec(allocation.requirement.kind())) { "mapped payload" } else { "owned" },
            "basis": "logical payload including padding/auxiliary storage; see placement for owned vs mapped backing; shared datasets counted once",
            "phase": if placement.options.contains_key(crate::placement::codec(allocation.requirement.kind())) { "mapped" } else { "resident" }, "source": base_source,
        }));
    }
    let cache_path = config
        .base_content_id
        .as_ref()
        .map(|_| cache_key.with_extension("cache.json"));
    #[cfg(feature = "benchmark")]
    let cache_path = if config.benchmark_trust_cache {
        config.benchmark_cache_file.clone().or(cache_path)
    } else {
        cache_path
    };
    let mut report = serde_json::json!({
        "dataset": config.dataset,
        "components": components,
        "query_count": query_count,
        "query_buffered_rows": if unknown_replay_buffers { None } else { Some(buffered) },
        "accounted_lower_bound_bytes": accounted,
        "build_lower_bound_bytes": build_lower_bound,
        "required_lower_bound_bytes": required,
        "budget_exceeded": exceeded,
        "num_points": num_points,
        "num_queries": query_count,
        "metadata_sources": {"base":base_source, "query":query_source, "graph":graph_source},
        "offline_assumptions": base_source == "configuration" || query_source == "configuration" || graph_source == "configuration",
        "unknown_replay_buffers": unknown_replay_buffers,
        "dimension": config.dimension,
        "base_format": config.base_format,
        "vector_storage": storage,
        "base_storage_bytes": base_bytes,
        "base_f32_bytes": if storage == VectorStorageKind::F32 { base_bytes } else { 0 },
        "base_u8_bytes": if storage == VectorStorageKind::U8 { base_bytes } else { 0 },
        "graph_slab_bytes": graph_bytes,
        "sidecar_bytes": sidecars.total_bytes(),
        "sidecar_owned_bytes": placement.resident_bytes
            - if placement.options.contains_key("base") { 0 } else { base_bytes }
            - if placement.options.contains_key("graph") { 0 } else { graph_bytes },
        "sidecar_mapped_bytes": placement.mapped_bytes - (base_bytes - owned_base) - (graph_bytes - owned_graph),
        "mapping_reserve_bytes": placement.reserve_bytes,
        "placement_limit_bytes": placement.limit_bytes,
        "storage_options": config.storage_options,
        "node_reader_bytes": node_reader_bytes,
        "resident_lower_bound_bytes": resident_lower_bound,
        "resident_lower_bound_gib": resident_lower_bound as f64 / BYTES_PER_GIB,
        "graph_source_policy": config.graph_source_policy,
        "graph_source": config.graph_source,
        "graph_selection_reason": config.graph_selection_reason,
        "graph_action": if cache_hit { "load cache" } else if config.graph_source == GraphSource::Rust { "build Rust graph" } else if staged_source.is_some() { "import PA staged export" } else if cfg!(feature = "parlay") { "build ParlayANN graph" } else { "PA input required" },
        "cache_hit": cache_hit,
        "cache_path": cache_path,
        "staged_source": staged_source,
        "memory_budget_gib": config.memory_budget_gib,
        "excludes": [
            "sidecar construction scratch",
            "query/result/GT container and allocator overhead beyond counted payloads",
            "query scratch",
            "allocator/runtime overhead",
            "in-process graph construction",
            "OS page cache"
        ],
        "note": "Lower bound only. Budget check is not a peak-RSS guarantee. \
                 Measure a smaller run before a full build. \
                 A PA cache miss requires an existing staged export."
    });
    report["mapped_bytes"] = placement.mapped_bytes.into();
    report["base_mapped"] = (owned_base == 0 && base_bytes != 0).into();
    report["graph_mapped"] = (owned_graph == 0 && graph_bytes != 0).into();
    Ok(report)
}

#[cfg(all(test, feature = "cli"))]
mod tests {
    use super::*;
    use crate::VectorFormat;

    fn native_config() -> ResolvedRunConfig {
        let mut config = crate::IndexOptions::default()
            .resolve(
                "missing.bvecs".into(),
                VectorFormat::Bvecs,
                128,
                1_000_000_000,
                None,
            )
            .unwrap();
        config.prepare_only = false;
        config.replay_queries = true;
        config.query = Some("missing-query.bvecs".into());
        config.groundtruth = Some("missing.ivecs".into());
        config.sweep.k = 10;
        config
    }
    #[test]
    fn offline_sweep_keeps_unknown_query_buffers_explicit() {
        let directory =
            std::env::temp_dir().join(format!("orion-offline-sweep-{}", std::process::id()));
        let mut config = native_config();
        config.base = directory.join("base.bvecs");
        config.query = Some(directory.join("query.bvecs"));
        config.staged_file = Some(directory.join("graph.staged"));
        config.cache_dir = directory.join("cache");
        config.num_queries = None;
        let report = preflight_report(&config).unwrap();
        assert_eq!(report["unknown_replay_buffers"], true);
        for name in ["query buffers", "result IDs", "ground truth IDs"] {
            let part = report["components"]
                .as_array()
                .unwrap()
                .iter()
                .find(|part| part["name"] == name)
                .unwrap();
            assert!(part["bytes"].is_null());
        }
        // The runtime path cannot substitute declarations for missing query files.
        assert!(estimate_resident_memory(&config, 1_000_000_000).is_err());
        config.num_queries = Some(10_000);
        let report = preflight_report(&config).unwrap();
        assert_eq!(report["unknown_replay_buffers"], false);
        assert_eq!(report["num_queries"], 10_000);
        config.prepare_only = true;
        let report = preflight_report(&config).unwrap();
        assert_eq!(report["num_queries"], 0);
    }

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
        let mut config = native_config();
        config.replay_queries = false;
        config.query = Some("-".into());
        config.staged_file = Some(staged);
        config.cache_dir = directory.clone();
        let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
        assert_eq!(report["base_f32_bytes"], 0);
        assert_eq!(report["base_u8_bytes"], 128_000_000_000u64);
        assert_eq!(report["sidecar_bytes"], 0);
        assert_eq!(report["resident_lower_bound_bytes"], 512_000_000_000u64);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "parlay")]
    #[test]
    fn slab_build_counts_graph_once() {
        let mut config = native_config();
        config.cache_dir =
            std::env::temp_dir().join(format!("orion-slab-budget-{}", std::process::id()));
        config.staged_file = None;
        config.staged_file_explicit = false;
        let report = preflight_report(&config).unwrap();
        assert_eq!(report["graph_action"], "build ParlayANN graph");
        assert_eq!(report["build_lower_bound_bytes"], 512_000_000_000u64);
        let working = report["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "ParlayANN working graph and extras")
            .unwrap();
        assert_eq!(working["bytes"], 0);
    }

    #[test]
    fn billion_lower_bound_and_budget_guard() {
        let mut config = crate::IndexOptions {
            storage: Some(VectorStorageKind::F32),
            graph: crate::GraphConfig {
                source: crate::GraphSourcePolicy::Rust,
                ..Default::default()
            },
            cache_dir: "/tmp/orion-resource-unit-test-unused".into(),
            ..Default::default()
        }
        .resolve(
            "missing-large.bvecs".into(),
            VectorFormat::Bvecs,
            128,
            1_000_000_000,
            None,
        )
        .unwrap();
        config.query = Some("-".into());
        config.groundtruth = Some("missing.ivecs".into());
        config.prepare_only = false;
        let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
        assert_eq!(report["base_f32_bytes"], 512_000_000_000u64);
        assert_eq!(report["graph_slab_bytes"], 384_000_000_000u64);
        assert_eq!(report["sidecar_bytes"], 128_000_000_000u64);
        config.memory_budget_gib = Some(128.);
        // Owned-only remains a strict guard. Auto can now move base and graph.
        config.storage_options.mode = crate::StorageMode::Resident;
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
            assert!(
                error.to_string().contains(component),
                "missing {component}: {error}"
            );
        }
        assert!(report["components"].as_array().unwrap().len() >= 10);
    }
}
