/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::report::table::BuildTiming;
use crate::runner::common::{AlgorithmRunner, SearchResult};
use diskann::index::{create_inmem_index, ANNInmemIndex};
use diskann::model::{IndexConfiguration, IndexWriteParametersBuilder};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use vector::Metric;

pub struct DiskANNRunner {
    index: Option<Box<dyn ANNInmemIndex<f32>>>,
    dimension: usize,
    search_list_size: u32,
    graph_degree: u32,
    alpha: f32,
    /// Temp file path for data (diskann-core requires file-based loading)
    temp_data_file: Option<PathBuf>,
    /// Mmap graph for zero-copy neighbor and vector access
    mmap_state: Option<DiskANNMmapState>,
}

/// State for mmap-based DiskANN search.
/// Graph and vectors are both read from the mmap file — no in-memory data arrays needed.
struct DiskANNMmapState {
    graph: platform::MmapGraph,
    start: u32,
}

impl DiskANNRunner {
    pub fn new(search_list_size: usize, graph_degree: u32, alpha: f32) -> Self {
        Self {
            index: None,
            dimension: 0,
            search_list_size: search_list_size as u32,
            graph_degree,
            alpha,
            temp_data_file: None,
            mmap_state: None,
        }
    }

    /// Write flat f32 data to a temp file in diskann binary format:
    /// [i32: num_points] [i32: dimension] [f32 * num_points * dimension]
    fn write_temp_data_file(
        data: &[f32],
        num_points: usize,
        dimension: usize,
    ) -> std::io::Result<PathBuf> {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("diskann_bench_{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path)?;

        file.write_all(&(num_points as i32).to_le_bytes())?;
        file.write_all(&(dimension as i32).to_le_bytes())?;

        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(byte_slice)?;
        file.flush()?;

        Ok(path)
    }
}

impl AlgorithmRunner for DiskANNRunner {
    fn name(&self) -> &str {
        "DiskANN (Vamana)"
    }

    fn build(&mut self, data: &[f32], num_points: usize, dimension: usize) -> BuildTiming {
        self.dimension = dimension;
        let start = Instant::now();

        // Write data to temp file (diskann-core requires file-based loading)
        let temp_path = Self::write_temp_data_file(data, num_points, dimension)
            .expect("Failed to write temp data file");
        self.temp_data_file = Some(temp_path.clone());

        // Use rayon thread count so enough query scratch objects are pre-allocated
        // for parallel search_batch (initialize_query_scratch creates 5 + num_threads).
        let num_threads = rayon::current_num_threads() as u32;
        let write_params =
            IndexWriteParametersBuilder::new(self.search_list_size, self.graph_degree)
                .with_alpha(self.alpha)
                .with_num_threads(num_threads)
                .build();

        let config = IndexConfiguration::new(
            Metric::L2,
            dimension,
            dimension, // aligned_dim = dim for standard dimensions
            num_points,
            false, // use_pq_dist
            0,     // num_pq_chunks
            false, // use_opq
            0,     // num_frozen_pts
            1.0,   // growth_potential
            write_params,
        );

        let mut index: Box<dyn ANNInmemIndex<f32>> =
            create_inmem_index(config).expect("Failed to create DiskANN index");

        index
            .build(temp_path.to_str().unwrap(), num_points)
            .expect("DiskANN build failed");

        let elapsed = start.elapsed();
        self.index = Some(index);
        BuildTiming {
            graph_build: elapsed,
            overhead: Duration::ZERO,
        }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchResult {
        // If mmap mode is active, use the standalone mmap search (reads vectors from mmap)
        if let Some(ref state) = self.mmap_state {
            let start = Instant::now();
            let results = match self.dimension {
                128 => {
                    let q = slice_to_array::<128>(query);
                    platform::mmap_greedy_search::<128>(
                        &state.graph,
                        &q,
                        state.start,
                        self.search_list_size as usize,
                        k,
                        Metric::L2,
                    )
                }
                960 => {
                    let q = slice_to_array::<960>(query);
                    platform::mmap_greedy_search::<960>(
                        &state.graph,
                        &q,
                        state.start,
                        self.search_list_size as usize,
                        k,
                        Metric::L2,
                    )
                }
                _ => panic!("Unsupported dimension for mmap search: {}", self.dimension),
            };
            let duration = start.elapsed();
            return SearchResult {
                neighbors: results.iter().map(|n| n.id).collect(),
                duration,
            };
        }

        let index = self.index.as_ref().expect("Index not built");
        let start = Instant::now();
        let mut indices = vec![0u32; k];
        index
            .search(query, k, self.search_list_size, &mut indices)
            .expect("DiskANN search failed");
        let duration = start.elapsed();
        SearchResult {
            neighbors: indices,
            duration,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }

    fn supports_mmap(&self) -> bool {
        true
    }

    fn save_mmap(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("diskann_graph.anns");
        let index = self.index.as_ref().expect("Index not built");
        // Save graph WITH vector data so mmap search can read vectors from disk
        index.save_graph_mmap_with_vectors(path.to_str().unwrap())?;
        Ok(path)
    }

    fn enable_mmap_search(&mut self, graph_path: &Path) -> anyhow::Result<()> {
        let start_node = self.index.as_ref().expect("Index not built").start_node();

        // Open mmap graph — vectors are included in the file, no in-memory data needed
        let mmap_graph = platform::MmapGraph::open(graph_path, 0)?;

        self.mmap_state = Some(DiskANNMmapState {
            graph: mmap_graph,
            start: start_node,
        });

        // Drop in-memory index to free heap
        self.index = None;
        Ok(())
    }
}

impl Drop for DiskANNRunner {
    fn drop(&mut self) {
        if let Some(ref path) = self.temp_data_file {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn slice_to_array<const N: usize>(slice: &[f32]) -> [f32; N] {
    assert!(slice.len() >= N);
    let mut arr = [0.0f32; N];
    arr.copy_from_slice(&slice[..N]);
    arr
}
