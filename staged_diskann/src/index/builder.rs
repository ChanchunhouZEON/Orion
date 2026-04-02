/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::index::{ANNInmemIndex, create_inmem_index};
use diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use diskann::model::{CsrGraph, FixedChunkPQTable, IndexConfiguration};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vector::Metric;

/// Re-export so downstream crates can use it without depending on diskann directly.
pub use diskann::model::CsrGraph as CsrGraphType;

/// Result of building a DiskANN index via diskann's parallel Vamana implementation.
///
/// `graph` is a lock-free CSR adjacency list extracted from the Vamana index.
/// `index` retains the original `InmemIndex` (behind trait object) so callers
/// can downcast to take the `InmemDataset` for zero-copy data sharing.
pub struct DiskANNBuildResult {
    /// Aligned CsrGraph with embedded bidir bitset per node.
    pub graph: CsrGraph,
    pub candidate_sets: Arc<Vec<Vec<u32>>>,
    pub entry_point: u32,
    /// The original index. Callers may downcast via `as_any_mut()` to take
    /// fields like `InmemDataset` before dropping.
    pub index: Box<dyn ANNInmemIndex<f32>>,
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub graph_build_time: Duration,
    pub pq_build_time: Duration,
}

/// Build a DiskANN index using diskann's optimized parallel Vamana build.
///
/// This replaces the old single-threaded `DiskANN::new()` from `diskann_base.rs`.
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
) -> diskann::common::ANNResult<DiskANNBuildResult> {

    // 1. Build Vamana graph via diskann (parallel, optimized)
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

    let mut index: Box<dyn ANNInmemIndex<f32>> =
        create_inmem_index::<f32>(config).expect("Failed to create diskann index");

    let graph_start = Instant::now();
    index
        .build_from_data(flat_data, num_points)
        .expect("Failed to build diskann index");
    let graph_build_time = graph_start.elapsed();

    let entry_point = index.start_node();

    let t_extract = Instant::now();
    let (graph, candidate_sets) = if compute_candidate_sets {
        let (g, cs) = index.extract_graph_and_candidates(num_points, graph_degree)?;
        (g, Arc::new(cs))
    } else {
        let (g, _) = index.extract_graph_and_candidates(num_points, graph_degree)?;
        (g, Arc::new(vec![Vec::new(); num_points]))
    };
    log::info!(
        "  extract_graph_and_candidates: {:.3}s",
        t_extract.elapsed().as_secs_f32()
    );

    // 3. Build PQ if requested
    let pq_start = Instant::now();
    let (pq, pq_codes) = if build_pq {
        let n_chunks = n_subquantizers.unwrap_or(8);
        let pq_table = FixedChunkPQTable::train(flat_data, num_points, dimension, n_chunks);
        let codes = pq_table.encode(flat_data, num_points);
        (Some(Arc::new(pq_table)), Some(codes))
    } else {
        (None, None)
    };
    let pq_build_time = pq_start.elapsed();

    Ok(DiskANNBuildResult {
        graph,
        candidate_sets,
        entry_point,
        index,
        pq,
        pq_codes,
        graph_build_time,
        pq_build_time,
    })
}
