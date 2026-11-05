/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::{Error, Result};
use orion_ann_diskann::index::{ANNInmemIndex, create_inmem_index};
use orion_ann_diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use orion_ann_diskann::model::{FixedChunkPQTable, IndexConfiguration};
use orion_ann_vector::Metric;
use std::sync::Arc;
#[cfg(feature = "benchmark")]
use std::time::Duration;
use std::time::Instant;

/// Owned Vamana build outputs. `index` retains the working graph/base; drop it
/// after extracting the partitions when preparing a separate Orion graph slab.
/// Optional PQ codes are training output, not a saved runtime snapshot.
pub struct DiskANNBuildResult {
    /// Per-node (local, remote, extra) partitions from extract.
    /// Empty if `compute_candidate_sets` was false.
    pub partitions: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    /// Entry point node ID.
    pub entry_point: u32,
    pub index: Box<dyn ANNInmemIndex<f32>>,
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    #[cfg(feature = "benchmark")]
    pub graph_build_time: Duration,
    #[cfg(feature = "benchmark")]
    pub pq_build_time: Duration,
}

/// Build a Vamana graph from finite row-major f32 vectors using squared L2.
///
/// # Example
/// ```no_run
/// use orion_ann_core::build_diskann_index;
/// let base: Vec<f32> = (0..128 * 100).map(|i| (i % 251) as f32).collect();
/// let built = build_diskann_index(&base, 100, 128, 1.2, 16, 64,
///     false, None, None, true, 16)?;
/// assert_eq!(built.partitions.len(), 100);
/// # Ok::<(), orion_ann_core::Error>(())
/// ```
///
/// # Errors
/// Rejects empty/inconsistent shape, non-finite coordinates, invalid graph bounds
/// and unsupported PQ parameters before calling the backend. Backend/I/O failures
/// retain their source chain. PQ uses 8-bit codes; chunks may have unequal widths.
///
/// # Ownership and resource use
/// Borrows input for this call, but the backend owns its copy. Candidate extraction
/// and optional PQ training need additional memory. Uses the current Rayon pool;
/// run inside a dedicated pool for a bounded worker count. It writes no graph file.
/// Allocation failures and internal backend invariant bugs follow the Rust host's
/// panic policy; this function does not catch panics. Callers needing full cache,
/// metric and budget validation should use `orion_ann::Index`.
#[allow(clippy::too_many_arguments)]
pub fn build_diskann_index(
    flat_data: &[f32],
    num_points: usize,
    dimension: usize,
    alpha: f32,
    graph_degree: u32,
    search_list_size: u32,
    build_pq: bool,
    n_subquantizers: Option<usize>,
    _n_bits: Option<u32>,
    compute_candidate_sets: bool,
    max_extra: usize,
) -> Result<DiskANNBuildResult> {
    let elements = num_points
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_argument("base shape overflows address space"))?;
    if num_points == 0
        || num_points >= u32::MAX as usize
        || dimension == 0
        || elements != flat_data.len()
        || flat_data.iter().any(|x| !x.is_finite())
    {
        return Err(Error::invalid_argument(
            "build requires finite, nonempty row-major vectors matching point count and dimension",
        )
        .at("build"));
    }
    if graph_degree == 0
        || graph_degree > u32::MAX - 4
        || search_list_size < graph_degree
        || !alpha.is_finite()
        || alpha < 1.0
        || max_extra > (u32::MAX - 4 - graph_degree) as usize
    {
        return Err(Error::invalid_argument(
            "invalid Vamana degree, search list, alpha or extra capacity",
        )
        .at("build"));
    }
    if build_pq
        && (n_subquantizers.unwrap_or(8) == 0
            || n_subquantizers.unwrap_or(8) > dimension
            || _n_bits.is_some_and(|bits| bits != 8))
    {
        return Err(
            Error::unsupported("PQ requires 1..=dimension chunks and 8-bit codes").at("build"),
        );
    }
    let num_threads = rayon::current_num_threads() as u32;
    let write_params = IndexWriteParametersBuilder::new(search_list_size, graph_degree)
        .with_alpha(alpha)
        .with_num_threads(num_threads)
        .with_compute_candidate_sets(compute_candidate_sets)
        .build();

    let config = IndexConfiguration::new(
        Metric::L2,
        dimension,
        dimension,
        num_points,
        false,
        0,
        false,
        0,
        1.0,
        write_params,
    );

    let mut index: Box<dyn ANNInmemIndex<f32>> = create_inmem_index::<f32>(config)?;

    #[cfg(feature = "benchmark")]
    let graph_start = Instant::now();
    index.build_from_data(flat_data, num_points)?;
    #[cfg(feature = "benchmark")]
    let graph_build_time = graph_start.elapsed();

    let entry_point = index.start_node();

    let t_extract = Instant::now();
    let partitions = if compute_candidate_sets {
        index.extract_graph_and_candidates(max_extra)?
    } else {
        vec![]
    };
    log::info!("  extract: {:.3}s", t_extract.elapsed().as_secs_f32());

    // Build PQ if requested.
    #[cfg(feature = "benchmark")]
    let pq_start = Instant::now();
    let (pq, pq_codes) = if build_pq {
        let n_chunks = n_subquantizers.unwrap_or(8);
        let pq_table = FixedChunkPQTable::train(flat_data, num_points, dimension, n_chunks);
        let codes = pq_table.encode(flat_data, num_points);
        (Some(Arc::new(pq_table)), Some(codes))
    } else {
        (None, None)
    };
    #[cfg(feature = "benchmark")]
    let pq_build_time = pq_start.elapsed();

    Ok(DiskANNBuildResult {
        partitions,
        entry_point,
        index,
        pq,
        pq_codes,
        #[cfg(feature = "benchmark")]
        graph_build_time,
        #[cfg(feature = "benchmark")]
        pq_build_time,
    })
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for DiskANNBuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskANNBuildResult")
            .field("partition_count", &self.partitions.len())
            .field("entry_point", &self.entry_point)
            .field("has_pq", &self.pq.is_some())
            .field("pq_code_bytes", &self.pq_codes.as_ref().map(Vec::len))
            .finish_non_exhaustive()
    }
}
