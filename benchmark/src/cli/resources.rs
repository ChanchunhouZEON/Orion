//! Header-only preflight: known resident components, not predicted peak RSS.
use super::{
    cache::CachePlan,
    config::{GraphSource, ResolvedRunConfig},
    data::inspect_vector_file,
};
use std::{io::Read, path::PathBuf};

const BYTES_PER_GIB: f64 = 1_073_741_824.0;
const GRAPH_HEADER_WORDS: u64 = 4;
const CACHE_LINE_BYTES: u64 = 64;
const SIMD_PAD_BYTES: u64 = 64;

/// Inspect input shapes and graph headers without allocating their payloads.
pub fn preflight_report(config: &ResolvedRunConfig) -> Result<serde_json::Value, String> {
    let base = inspect_vector_file(&config.base, config.base_format)?;
    let query = inspect_vector_file(&config.query, config.query_format)?;

    if base.dimension != config.dimension || query.dimension != base.dimension {
        return Err("base/query/config dimension mismatch".into());
    }

    estimate_resident_memory(config, base.count.min(config.max_points))
}

fn graph_stride_bytes(max_degree: u64, max_extra: u64) -> u64 {
    // Each node stores header words and both adjacency regions, rounded to a cache line.
    let record_bytes = (GRAPH_HEADER_WORDS + max_degree + max_extra) * 4;
    (record_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES
}

/// Prefer actual graph/export capacity over configuration defaults. STAG's
/// max-extra can differ from the requested value and determines slab allocation.
/// Returns bytes per node and the export path, when an import is required.
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

fn checked_bytes(count: u64, bytes_per_item: u64) -> Result<u64, String> {
    count
        .checked_mul(bytes_per_item)
        .ok_or_else(|| "memory estimate overflow".into())
}

fn estimate_resident_memory(
    config: &ResolvedRunConfig,
    num_points: usize,
) -> Result<serde_json::Value, String> {
    if num_points == 0 || num_points > u32::MAX as usize || config.sweep.k > num_points {
        return Err("invalid point count or k".into());
    }

    let cache = CachePlan::resolve(config, num_points)?;
    let (stride_bytes, staged_source) = resolve_graph_layout(config, &cache, num_points)?;

    // Byte inputs are expanded to f32 in the resident index. SIMD padding is
    // appended once to the allocation, not to every vector.
    let coordinate_count = checked_bytes(num_points as u64, config.dimension as u64)?;
    let base_payload_bytes = checked_bytes(coordinate_count, 4)?;
    let base_bytes = base_payload_bytes
        .checked_add(SIMD_PAD_BYTES)
        .ok_or("memory estimate overflow")?;
    let graph_bytes = checked_bytes(num_points as u64, stride_bytes)?;
    let node_reader_bytes = checked_bytes(num_points as u64, std::mem::size_of::<usize>() as u64)?;

    // Only the L2U8 representation is accounted for here. Other cascade buffers
    // remain explicitly excluded, rather than presented as a complete estimate.
    let admission_bytes = if matches!(
        config.cascade.admission,
        crate::cascade::AdmissionChoice::L2U8
    ) {
        let aligned_dimension = (config.dimension as u64 + 31) / 32 * 32;
        checked_bytes(num_points as u64, aligned_dimension)?
    } else {
        0
    };

    let resident_lower_bound = base_bytes
        .checked_add(admission_bytes)
        .and_then(|bytes| bytes.checked_add(graph_bytes))
        .and_then(|bytes| bytes.checked_add(node_reader_bytes))
        .ok_or("memory estimate overflow")?;

    if config
        .memory_budget_gib
        .is_some_and(|budget| resident_lower_bound as f64 > budget * BYTES_PER_GIB)
    {
        return Err(format!(
            "known base+graph+known admission memory alone requires {:.2} GiB, \
             above --memory-budget-gib; sidecars/build scratch require additional memory",
            resident_lower_bound as f64 / BYTES_PER_GIB
        ));
    }

    // Retain output field names for existing run-log consumers.
    Ok(serde_json::json!({
        "num_points": num_points,
        "dimension": config.dimension,
        "base_format": config.base_format,
        "base_f32_bytes": base_bytes,
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
            "queries and ground truth",
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
    log::info!("Resource preflight: {report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn billion_lower_bound_and_budget_guard() {
        let args = super::super::config::Args::try_parse_from([
            "orion",
            "--base",
            "missing-large.bvecs",
            "--query",
            "missing-query.bvecs",
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
        assert!(estimate_resident_memory(&config, 1_000_000_000).is_err());
    }
}
