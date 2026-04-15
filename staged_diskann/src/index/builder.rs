/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::index::{ANNInmemIndex, create_inmem_index};
use diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use diskann::model::{FixedChunkPQTable, IndexConfiguration};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vector::Metric;

/// Result of building a DiskANN index.
///
/// `graph` is the final InMemoryGraph with neighbors distance-sorted in-place.
/// `candidate_sets` are the per-node candidate sets (key_nbrs ∪ pruned points).
pub struct DiskANNBuildResult {
    /// Per-node (local, remote, extra) partitions from extract.
    /// Empty if `compute_candidate_sets` was false.
    pub partitions: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    /// Entry point node ID.
    pub entry_point: u32,
    pub index: Box<dyn ANNInmemIndex<f32>>,
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub graph_build_time: Duration,
    pub pq_build_time: Duration,
}

/// Build a DiskANN Vamana index.
///
/// When `compute_candidate_sets` is true, neighbors are distance-sorted in-place,
/// the slab is enriched via single-pass prune, and candidate_sets are extracted.
/// Build a DiskANN Vamana index.
///
/// When `compute_candidate_sets` is true, extracts per-node (local, remote, extra)
/// partitions for PhasedGraph construction. `max_extra` controls how many
/// candidates from the merged remote+extra set are kept per node.
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
) -> diskann::common::ANNResult<DiskANNBuildResult> {
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
    let partitions = if compute_candidate_sets {
        index.extract_graph_and_candidates(max_extra)?
    } else {
        vec![]
    };
    log::info!("  extract: {:.3}s", t_extract.elapsed().as_secs_f32());

    // Build PQ if requested.
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
        partitions,
        entry_point,
        index,
        pq,
        pq_codes,
        graph_build_time,
        pq_build_time,
    })
}
