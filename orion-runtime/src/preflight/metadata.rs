/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Resolve vector counts and graph layout from validated headers or declarations.
//! Payload allocation and placement are deliberately absent from this module.
use crate::{
    config::{GraphSource, ResolvedRunConfig},
    data::inspect_vector_file,
    graph::cache::CachePlan,
    Error, Result,
};
use std::{io::Read, path::PathBuf};

const GRAPH_HEADER_WORDS: u64 = 4;
const CACHE_LINE_BYTES: u64 = 64;

/// Resolved counts and graph layout consumed by the pure budget calculator.
/// All filesystem observations are resolved before placement arithmetic begins.
pub(super) struct MemoryInputs {
    pub(super) num_points: usize,
    pub(super) base_source: &'static str,
    pub(super) query_count: Option<usize>,
    pub(super) query_source: &'static str,
    pub(super) stride_bytes: u64,
    pub(super) graph_source: &'static str,
    pub(super) staged_source: Option<PathBuf>,
    pub(super) cache_hit: bool,
    pub(super) cache_key: PathBuf,
}

/// Missing files are allowed only during offline planning. Existing invalid files
/// always fail, so a declaration cannot hide corruption or a mismatched dataset.
pub(super) fn resolve_vector_count(
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
                    "{}: file count {} != declared count {}; update the declared count or explicitly override the input path",
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
pub(super) fn resolve_memory_inputs(
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

pub(super) fn inputs_with_plan(
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
    record_bytes.div_ceil(CACHE_LINE_BYTES) * CACHE_LINE_BYTES
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
