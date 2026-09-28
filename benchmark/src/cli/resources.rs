//! Header-only preflight. These are lower bounds, not a prediction of build peak RSS.
use super::{
    cache::CachePlan,
    config::{GraphSource, ResolvedRunConfig},
    data::inspect,
};
use std::io::Read;

pub fn report(config: &ResolvedRunConfig) -> Result<serde_json::Value, String> {
    let base = inspect(&config.base, config.base_format)?;
    let query = inspect(&config.query, config.query_format)?;
    if base.dimension != config.dimension || query.dimension != base.dimension {
        return Err("base/query/config dimension mismatch".into());
    }
    estimate(config, base.count.min(config.max_points))
}

fn estimate(config: &ResolvedRunConfig, n: usize) -> Result<serde_json::Value, String> {
    if n == 0 || n > u32::MAX as usize || config.sweep.k > n {
        return Err("invalid point count or k".into());
    }
    let cache = CachePlan::select(config, n)?;
    let mut degree = config.graph_degree as u64;
    let mut extra = config.max_extra as u64;
    let mut stride = ((4 + degree + extra) * 4 + 63) / 64 * 64;
    let mut source = None;
    if cache.hit {
        let mut h = [0; 16];
        std::fs::File::open(cache.path.with_extension("pgraph"))
            .and_then(|mut f| f.read_exact(&mut h))
            .map_err(|e| e.to_string())?;
        stride = u32::from_le_bytes(h[8..12].try_into().unwrap()) as u64 * 4;
    } else if config.graph_source == GraphSource::Parlayann {
        let path = CachePlan::source_for_import(config, n)?;
        let mut h = [0; 24];
        std::fs::File::open(&path)
            .and_then(|mut f| f.read_exact(&mut h))
            .map_err(|e| e.to_string())?;
        degree = u32::from_le_bytes(h[12..16].try_into().unwrap()) as u64;
        extra = u32::from_le_bytes(h[16..20].try_into().unwrap()) as u64;
        if degree != config.graph_degree as u64 || degree > n as u64 || extra > n as u64 {
            return Err("staged capacities/config degree mismatch".into());
        }
        stride = ((4 + degree + extra) * 4 + 63) / 64 * 64;
        source = Some(path);
    }
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or("memory estimate overflow");
    let base_bytes = mul(mul(n as u64, config.dimension as u64)?, 4)?
        .checked_add(64)
        .ok_or("memory estimate overflow")?;
    let graph_bytes = mul(n as u64, stride)?;
    let readers = mul(n as u64, std::mem::size_of::<usize>() as u64)?;
    let admission_bytes = if matches!(
        config.cascade.admission,
        crate::cascade::AdmissionChoice::L2U8
    ) {
        mul(n as u64, (config.dimension as u64 + 31) / 32 * 32)?
    } else {
        0
    };
    let minimum = base_bytes
        .checked_add(admission_bytes)
        .and_then(|x| x.checked_add(graph_bytes))
        .and_then(|x| x.checked_add(readers))
        .ok_or("memory estimate overflow")?;
    if config
        .memory_budget_gib
        .is_some_and(|g| minimum as f64 > g * 1073741824.)
    {
        return Err(format!("known base+graph+known admission memory alone requires {:.2} GiB, above --memory-budget-gib; sidecars/build scratch require additional memory", minimum as f64 / 1073741824.));
    }
    Ok(serde_json::json!({
        "num_points": n, "dimension": config.dimension,
        "base_format": config.base_format, "base_f32_bytes": base_bytes,
        "graph_slab_bytes": graph_bytes, "l2_u8_admission_bytes": admission_bytes, "node_reader_bytes": readers,
        "resident_lower_bound_bytes": minimum,
        "resident_lower_bound_gib": minimum as f64 / 1073741824.,
        "cache_hit": cache.hit, "cache_path": cache.path, "staged_source": source,
        "memory_budget_gib": config.memory_budget_gib,
        "excludes": ["sidecars other than L2U8 admission, and sidecar construction overhead", "queries and ground truth", "query scratch", "allocator/runtime overhead", "in-process graph construction", "OS page cache"],
        "note": "Lower bound only. Budget check is not a peak-RSS guarantee. Measure a smaller run before a full build. A PA cache miss requires an existing staged export."
    }))
}

pub fn check(config: &ResolvedRunConfig, n: usize) -> Result<(), String> {
    let report = estimate(config, n)?;
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
        let report = estimate(&config, 1_000_000_000).unwrap();
        assert_eq!(report["base_f32_bytes"], 512_000_000_064u64);
        assert_eq!(report["graph_slab_bytes"], 384_000_000_000u64);
        assert_eq!(report["l2_u8_admission_bytes"], 128_000_000_000u64);
        config.memory_budget_gib = Some(128.);
        assert!(estimate(&config, 1_000_000_000).is_err());
    }
}
