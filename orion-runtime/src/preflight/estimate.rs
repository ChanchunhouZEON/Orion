/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Combine persistent payloads, query buffers and placement into a report.
//! Header inspection lives in metadata; enforcement and table formatting are separate.
use super::{metadata::MemoryInputs, BYTES_PER_GIB};
use crate::{
    config::{GraphSource, ResolvedRunConfig, VectorStorageKind},
    Error, Result,
};

/// Multiplies payload dimensions without allowing an overflow to understate the memory budget.
pub(super) fn checked_mul_bytes(count: u64, bytes_per_item: u64) -> Result<u64> {
    count
        .checked_mul(bytes_per_item)
        .ok_or_else(|| "memory estimate overflow".into())
}

/// Pure accounting shared by offline planning and strict runtime budget checks.
pub(super) fn calculate_memory(
    config: &ResolvedRunConfig,
    input: MemoryInputs,
) -> Result<serde_json::Value> {
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
            orion_ann_vector::IpU8Distance,
        >(config.dimension)?,
    };
    let base_bytes = checked_mul_bytes(num_points as u64, layout.stride_bytes() as u64)?;
    let graph_bytes = checked_mul_bytes(num_points as u64, stride_bytes)?;
    // Graph storage is immutable during search; no per-node reader array is allocated.
    let node_reader_bytes = 0u64;

    let sidecar_requirements = config.run_phase().sidecar_requirements(config.index_plan());
    let norm_bytes = crate::placement::native_norm_bytes(config, num_points)?;
    let sidecars = crate::sidecars::requirements::estimate_sidecars(
        orion_ann_core::QuantizedDatasetShape {
            rows: num_points,
            dimension: config.dimension,
        },
        &sidecar_requirements,
    )?;

    let super::queries::QueryBuffers {
        buffered,
        query_bytes,
        result_bytes,
        gt_bytes,
        unknown_replay: unknown_replay_buffers,
    } = super::queries::QueryBuffers::estimate(config, query_count, layout)?;
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
        &sidecar_requirements,
        base_bytes,
        graph_bytes,
        fixed_bytes - base_bytes - graph_bytes,
        orion_ann_platform::resources::MemoryResources::default(),
    )?;
    let resident_lower_bound = placement.resident_bytes;
    let accounted = resident_lower_bound
        .checked_add(query_bytes)
        .and_then(|v| v.checked_add(result_bytes))
        .and_then(|v| v.checked_add(gt_bytes))
        .ok_or("memory estimate overflow: query/result/GT buffers")?;
    // ParlayANN borrows the final slab for both adjacency and extras. There is
    // no independent N-sized graph/extras allocation in the production path.
    // Sidecars are prepared later; candidate/statistics scratch is listed below.
    // Parlay cosine owns temporary norms while constructing; they are dropped
    // before preparing the separately placed search norm object.
    let working_bytes =
        if !cache_hit && staged_source.is_none() && config.graph_source == GraphSource::Parlayann {
            norm_bytes
        } else {
            0
        };
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
        {"name":"ParlayANN construction norms", "bytes":working_bytes, "basis":"temporary native-cosine squared norms (u32/u64), released before search norms; adjacency and extras share the final graph slab", "phase":"build/import"},
        {"name":"graph construction temporary memory", "bytes":null, "basis":if cache_hit {"cache load; decoder and allocator overhead not estimated"} else if config.graph_source == GraphSource::Supplied {"streamed supplier import; one bounded partition row and validation scratch"} else if config.graph_source == GraphSource::Parlayann {
            if staged_source.is_some() { "streamed STAG import; no in-process builder" }
            else { "ParlayANN borrows base and graph slab; construction statistics, insertion batches and worker candidate/partition scratch remain additional" }
        } else {"Rust builder retains working dataset and graph partitions; peak not estimated"}, "phase":"build/import"},
        {"name":"allocator/runtime/OS overhead", "bytes":null, "basis":"not included in payload bounds", "phase":"all"}
    ]);

    // Retain output field names for existing run-log consumers.
    for part in components.as_array_mut().unwrap() {
        part["source"] = serde_json::json!(match part["name"].as_str().unwrap() {
            "base vectors" | "node reader counters" => base_source,
            "graph slab" => graph_source,
            "ParlayANN construction norms" => "configuration",
            "query buffers" | "result IDs" | "ground truth IDs" => query_source,
            _ => "not estimated",
        });
    }
    // Dataset owners supply all payload components; stage labels never affect
    // allocation identity or accounting.
    if norm_bytes != 0 {
        let mapped = placement
            .options
            .contains_key(orion_ann_core::model::dataset::native_u8::NORM_CODEC);
        components.as_array_mut().unwrap().push(serde_json::json!({"name":"native u8 squared norms","bytes":norm_bytes,"placement":if mapped {"mapped payload"}else{"owned"},"basis":"one exact u32 squared norm per vector; u64 when 65025 × dimension exceeds u32","phase":if mapped {"mapped"}else{"resident"},"source":base_source}));
    }
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
        "sidecar_bytes": sidecars.total_bytes() as u64 + norm_bytes,
        "sidecar_owned_bytes": placement.resident_bytes
            - if placement.options.contains_key("base") { 0 } else { base_bytes }
            - if placement.options.contains_key("graph") { 0 } else { graph_bytes },
        "sidecar_mapped_bytes": placement.mapped_bytes - (base_bytes - owned_base) - (graph_bytes - owned_graph),
        "placement_limit_bytes": placement.limit_bytes,
        "storage_options": config.storage_options,
        "node_reader_bytes": node_reader_bytes,
        "resident_lower_bound_bytes": resident_lower_bound,
        "resident_lower_bound_gib": resident_lower_bound as f64 / BYTES_PER_GIB,
        "graph_source_policy": config.graph_source_policy,
        "graph_source": config.graph_source,
        "graph_selection_reason": config.graph_selection_reason,
        "graph_action": if cache_hit { "load cache" } else if config.graph_source == GraphSource::Supplied { "import supplier partitions" } else if config.graph_source == GraphSource::Rust { "build Rust graph" } else if staged_source.is_some() { "import PA staged export" } else if cfg!(feature = "parlay") { "build ParlayANN graph" } else { "PA input required" },
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
        "note": "Lower bound only; not a peak-RSS guarantee. Mapped bytes need backing storage, \
                 whose free space and quota are not checked here. Graph-only does not validate \
                 search sidecars. Explicit PA exports remain required; an enabled native builder \
                 may build a missing graph when no export was explicitly requested."
    });
    report["planned_phase"] = if config.prepare_only {
        "graph-only"
    } else {
        "search-ready"
    }
    .into();
    report["validation_scope"] = serde_json::json!({
        "resident_payload_budget": if placement.limit_bytes.is_some() { "checked lower bound" } else { "estimate only; no explicit limit" },
        "mmap_payload_limit": "not configured; no separate limit API",
        "mmap_backing_capacity": "not checked; filesystem quota may differ from free space",
        "peak_rss": "not bounded; mapped pages and unknown scratch can occupy RAM"
    });
    report["placement_objects"] = serde_json::to_value(&placement.objects)?;
    report["inactive_placement_overrides"] = serde_json::to_value(&placement.inactive_overrides)?;
    report["metric"] = serde_json::json!(config.metric);
    report["cascade"] = serde_json::json!(config.cascade());
    report["search_dispatch"] = config
        .cascade()
        .dispatch_description(config.storage())
        .into();
    report["native_norm_bytes"] = norm_bytes.into();
    report["mapped_bytes"] = placement.mapped_bytes.into();
    report["base_mapped"] = (owned_base == 0 && base_bytes != 0).into();
    report["graph_mapped"] = (owned_graph == 0 && graph_bytes != 0).into();
    Ok(report)
}
