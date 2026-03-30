/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::index::{ANNInmemIndex, create_inmem_index};
use diskann::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use diskann::model::{FixedChunkPQTable, InMemoryGraph, IndexConfiguration};
use ndarray::ArcArray2;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vector::Metric;

/// Result of building a DiskANN index via diskann's parallel Vamana implementation.
pub struct DiskANNBuildResult {
    pub graph: InMemoryGraph,
    pub candidate_sets: Arc<Vec<HashSet<u32>>>,
    pub entry_point: u32,
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
    data: &ArcArray2<f32>,
    alpha: f32,
    graph_degree: u32,
    search_list_size: u32,
    build_pq: bool,
    n_subquantizers: Option<usize>,
    _n_bits: Option<u32>,
    compute_candidate_sets: bool,
) -> DiskANNBuildResult {
    let num_points = data.nrows();
    let dimension = data.ncols();

    // 1. Build Vamana graph via diskann (parallel, optimized)
    let write_params = IndexWriteParametersBuilder::new(search_list_size, graph_degree)
        .with_alpha(alpha)
        .with_num_threads(0) // use all cores
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

    // Flatten the ndarray data into a contiguous slice
    let flat_data: Vec<f32> = if let Some(slice) = data.as_slice() {
        slice.to_vec()
    } else {
        data.iter().copied().collect()
    };

    let graph_start = Instant::now();
    index
        .build_from_data(&flat_data, num_points)
        .expect("Failed to build diskann index");
    let graph_build_time = graph_start.elapsed();

    let entry_point = index.start_node();

    // 2. Extract candidate sets
    let candidate_sets = if compute_candidate_sets {
        Arc::new(
            index
                .extract_candidate_sets()
                .unwrap_or_else(|| vec![HashSet::new(); num_points]),
        )
    } else {
        Arc::new(vec![HashSet::new(); num_points])
    };

    // 3. Extract graph
    let graph_map = index.extract_graph();
    let graph = InMemoryGraph::from_hashmap(&graph_map, num_points, graph_degree);

    // 4. Build PQ if requested — using FixedChunkPQTable
    let pq_start = Instant::now();
    let (pq, pq_codes) = if build_pq {
        let n_chunks = n_subquantizers.unwrap_or(8);
        let pq_table = FixedChunkPQTable::train(&flat_data, num_points, dimension, n_chunks);
        let codes = pq_table.encode(&flat_data, num_points);
        (Some(Arc::new(pq_table)), Some(codes))
    } else {
        (None, None)
    };
    let pq_build_time = pq_start.elapsed();

    DiskANNBuildResult {
        graph,
        candidate_sets,
        entry_point,
        pq,
        pq_codes,
        graph_build_time,
        pq_build_time,
    }
}
