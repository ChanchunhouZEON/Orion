/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::{CandidateSetManager, Neighbor, NeighborPriorityQueue};
use crate::pq::ProductQuantizer;
#[cfg(feature = "indicatif")]
use crate::utils::NODES_PROGRESS_STYLE;
use crate::utils::{DELIMITER_LENGTH, l2_distance};
use diskann::common::{ANNError, ANNResult};
use diskann::model::InMemoryGraph;
use diskann::model::graph::AdjacencyList;
#[cfg(feature = "indicatif")]
use indicatif::ProgressFinish::AndLeave;
#[cfg(feature = "indicatif")]
use indicatif::ProgressIterator;

use ndarray::{ArcArray2, ArrayView1};
use rand::RngExt;
use rand::rngs::ThreadRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vector::{FullPrecisionDistance, Metric};

/// Serializable on-disk representation of DiskANN graph metadata.
#[derive(Serialize, Deserialize)]
struct DiskANNOnDisk {
    alpha: f32,
    graph_degree: usize,
    search_list_size: usize,
    entry_point: u32,
    graph: HashMap<u32, Vec<u32>>,
    candidate_set_manager: CandidateSetManager,
    use_pq: bool,
    graph_save_path: PathBuf,
}

/// Base DiskANN (Vamana) graph index with const-generic dimension for SIMD search.
pub struct DiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pub alpha: f32,
    pub graph_degree: usize,
    pub search_list_size: usize,
    pub entry_point: u32,
    pub graph: InMemoryGraph,
    pub candidate_set_manager: CandidateSetManager,
    pub use_pq: bool,
    pub graph_save_path: PathBuf,
    pub build_candidate_sets: bool,

    /// ndarray data for build paths (dynamic dimension)
    pub data: ArcArray2<f32>,
    /// Fixed-size arrays for SIMD-accelerated search
    pub data_arrays: Vec<[f32; N]>,
    pub pq: Option<ProductQuantizer>,
    pub pq_save_path: Option<PathBuf>,
    pub is_save: bool,
}

impl<const N: usize> DiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: ArcArray2<f32>,
        alpha: f32,
        graph_degree: usize,
        search_list_size: usize,
        graph_save_path: Option<PathBuf>,
        use_pq: bool,
        pq_save_path: Option<PathBuf>,
        n_subquantizers: Option<usize>,
        n_bits: Option<u32>,
        is_save: bool,
        build_candidate_sets: bool,
    ) -> Self {
        assert_eq!(
            data.ncols(),
            N,
            "Data dimension {} does not match const generic N={}",
            data.ncols(),
            N
        );

        let n = data.nrows();
        let entry_point = rand::random_range(0..n) as u32;

        let graph_save_path = graph_save_path.unwrap_or_else(|| {
            let dir = PathBuf::from("diskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!(
                "dskann_graph_g{}_L{}.bin",
                graph_degree, search_list_size
            ))
        });

        let pq = if use_pq {
            let n_sub = n_subquantizers.unwrap_or(8);
            let bits = n_bits.unwrap_or(8);
            Some(ProductQuantizer::new(n, data.ncols(), n_sub, bits))
        } else {
            None
        };

        let pq_save_path = if use_pq {
            Some(pq_save_path.unwrap_or_else(|| {
                let dir = PathBuf::from("pq");
                fs::create_dir_all(&dir).unwrap();
                dir.join(format!(
                    "pq_n{}_dim{}_subquantizers{}_bits{}.bin",
                    data.nrows(),
                    data.ncols(),
                    n_subquantizers.unwrap_or(8),
                    n_bits.unwrap_or(8)
                ))
            }))
        } else {
            None
        };

        let graph = InMemoryGraph::new(n, graph_degree as u32);

        let candidate_set_manager = if build_candidate_sets {
            CandidateSetManager::with_anchor_sets(n)
        } else {
            CandidateSetManager::default()
        };

        let mut diskann = Self {
            alpha,
            graph_degree,
            search_list_size,
            entry_point,
            graph,
            candidate_set_manager,
            use_pq,
            graph_save_path,
            build_candidate_sets,
            data,
            data_arrays: Vec::new(),
            pq,
            pq_save_path,
            is_save,
        };

        diskann.load_or_build_optimized_graph();
        diskann.convert_data_for_search();

        if use_pq {
            diskann.load_or_build_pq();
        }

        diskann
    }

    /// Convert ndarray data rows to fixed-size arrays for SIMD search.
    pub fn convert_data_for_search(&mut self) {
        let n = self.data.nrows();
        self.data_arrays = Vec::with_capacity(n);
        for i in 0..n {
            let row = self.data.row(i);
            let mut arr = [0.0f32; N];
            arr.copy_from_slice(row.as_slice().unwrap());
            self.data_arrays.push(arr);
        }
    }

    fn load_or_build_optimized_graph(&mut self) {
        log::info!(
            "🏎️: trying to load the graph structure from '{}'.",
            self.graph_save_path.display()
        );

        if self.load(&self.graph_save_path.clone()).is_err() {
            log::info!("🚫: failed to load the graph structure.");
            log::info!("🏗️: building and optimizing graph structure.");
            let start_time = Instant::now();

            // 1. Build initial graph
            log::info!("🕸️: initializing regular graph structure.");
            let _ = self.build_initial_graph().map_err(|_| {
                ANNError::log_lock_poison_error("Unable to build initial graph".to_string())
            });

            // 2. Perform robust pruning
            log::info!("✂️: robust pruning.");
            let _ = self.robust_prune().map_err(|_| {
                ANNError::log_lock_poison_error("Unable to build saturate graph".to_string())
            });

            log::info!(
                "⌛️: total time in building optimized graph is {:.2} seconds",
                start_time.elapsed().as_secs_f32()
            );

            if self.build_candidate_sets {
                self.augment_candidate_sets()
                    .expect("Augment candidate sets failed");
            }

            if self.is_save {
                self.save(&self.graph_save_path.clone()).unwrap();
            }
        } else {
            log::info!("⌛️: existing optimized graph structure loaded.");
        }
        log::info!("{}", "-".repeat(DELIMITER_LENGTH));
    }

    fn load_or_build_pq(&mut self) {
        if let Some(ref pq_path) = self.pq_save_path {
            let loaded = ProductQuantizer::load_model(pq_path);
            if loaded.is_ok() && loaded.as_ref().unwrap() == self.pq.as_ref().unwrap() {
                self.pq = Some(loaded.unwrap());
                return;
            }
        }

        log::info!("🏗️: building product quantum for raw data.");
        let start_time = Instant::now();
        let pq = self.pq.as_mut().unwrap();
        pq.train(self.data.clone());
        pq.encode(self.data.clone());
        log::info!(
            "⌛️: building product quantum total time: {:.2} seconds.",
            start_time.elapsed().as_secs_f32()
        );
        log::info!("{}", "-".repeat(DELIMITER_LENGTH));

        if self.is_save {
            if let Some(ref pq_path) = self.pq_save_path {
                pq.save_model(pq_path).unwrap();
            }
        }
    }

    // ─── Builder: uses ndarray l2_distance (dynamic dimension ok for build) ───

    fn build_initial_graph_inner(&self, rng: &mut ThreadRng, i: usize, n: usize) -> ANNResult<()> {
        let mut neighbors = HashSet::new();
        while neighbors.len() < self.graph_degree {
            let j = rng.random_range(0..n);
            if j != i {
                neighbors.insert(j as u32);
            }
        }
        self.graph
            .set_neighbors_from_vec(i as u32, neighbors.into_iter().collect())?;

        Ok(())
    }

    fn build_initial_graph(&mut self) -> ANNResult<()> {
        let n = self.data.nrows();
        assert!(
            self.graph_degree < n,
            "Graph degree must be less than the number of points"
        );

        #[cfg(feature = "indicatif")]
        {
            let style = NODES_PROGRESS_STYLE.clone();
            (0..n)
                .progress_with_style(style)
                .with_finish(AndLeave)
                .try_for_each(|i| {
                    let mut rng = rand::rng();
                    self.build_initial_graph_inner(&mut rng, i, n)
                })?;
        }

        #[cfg(not(feature = "indicatif"))]
        {
            (0..n).into_par_iter().try_for_each_init(
                || rand::rng(),
                |rng, i| self.build_initial_graph_inner(rng, i, n),
            )?;
        }

        Ok(())
    }

    fn robust_prune_inner(&self, i: usize) -> ANNResult<()> {
        let greedy = self.greedy_search_for_candidates(i as u32)?;

        let new_neighbors = self.prune_node_neighbors(i as u32, greedy);

        for &v in new_neighbors.iter() {
            let mut entry_neighbors_guard = self.graph.write_vertex_and_neighbors(v)?;
            let entry_neighbors = entry_neighbors_guard.get_neighbors();
            if !entry_neighbors.contains(&(i as u32)) {
                if entry_neighbors.len() < self.graph_degree {
                    let mut updated = Vec::with_capacity(self.graph_degree);
                    updated.extend_from_slice(entry_neighbors.as_slice());
                    updated.push(i as u32);
                    entry_neighbors_guard.set_neighbors(AdjacencyList::from(updated));
                } else {
                    let mut candidates = entry_neighbors.clone().to_vec();
                    candidates.push(i as u32);
                    let new_neighbor_neighbors = self.prune_node_neighbors(v, candidates);
                    entry_neighbors_guard
                        .set_neighbors(AdjacencyList::from(new_neighbor_neighbors));
                }
            }
        }

        self.graph.set_neighbors_from_vec(i as u32, new_neighbors)?;
        Ok(())
    }

    fn robust_prune(&mut self) -> ANNResult<()> {
        let n = self.data.nrows();

        #[cfg(feature = "indicatif")]
        {
            let style = NODES_PROGRESS_STYLE.clone();
            (0..n)
                .progress_with_style(style)
                .with_finish(AndLeave)
                .try_for_each(|i| self.robust_prune_inner(i))?;
        }

        #[cfg(not(feature = "indicatif"))]
        {
            (0..n)
                .into_par_iter()
                .try_for_each(|i| self.robust_prune_inner(i))?;
        }

        Ok(())
    }

    fn prune_node_neighbors(&self, node: u32, candidates: Vec<u32>) -> Vec<u32> {
        let base = self.data.row(node as usize);
        let mut pruned_pairs: hashbrown::HashMap<u32, Vec<u32>> = hashbrown::HashMap::new();

        let mut dist: Vec<_> = candidates
            .into_iter()
            .filter(|&v| v != node)
            .map(|v| {
                let d = l2_distance(base, self.data.row(v as usize));
                (d, v)
            })
            .collect();

        dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut selected = Vec::new();
        let mut deleted = HashSet::new();

        for (i, &(d, u)) in dist.iter().enumerate() {
            if selected.len() >= self.graph_degree {
                break;
            }
            if deleted.contains(&u) {
                continue;
            }
            selected.push(u);

            for &(_, v) in &dist[i + 1..] {
                let duv = l2_distance(self.data.row(u as usize), self.data.row(v as usize));
                if duv * self.alpha <= d {
                    deleted.insert(v);
                    pruned_pairs.entry(u).or_default().push(v);
                }
            }
        }

        if self.build_candidate_sets && !pruned_pairs.is_empty() {
            if let Some(ref cas) = self.candidate_set_manager.candidate_anchor_sets {
                for (anchor, pruned) in pruned_pairs.iter() {
                    if let Some(slot) = cas.get(*anchor as usize) {
                        let mut map = slot.lock().unwrap();
                        map.entry(node).or_default().extend(pruned.iter().copied());
                    }
                }
            }
        }

        let degree = self.graph_degree.min(selected.len());

        let mut result = Vec::with_capacity(self.graph_degree);

        result.extend_from_slice(&selected[..degree]);

        result
    }

    fn augment_candidate_sets(&mut self) -> ANNResult<()> {
        let anchor_sets = self
            .candidate_set_manager
            .candidate_anchor_sets
            .as_ref()
            .ok_or_else(|| {
                ANNError::log_candidate_anchor_sets_error(
                    "Candidate anchor sets is NULL.".to_string(),
                )
            })?;
        self.candidate_set_manager
            .candidate_sets
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(origin, candidates)| {
                let origin = origin as u32;
                let mut target_set = HashSet::<u32>::new();
                target_set.insert(origin);

                let origin_neighbors_guard = self.graph.read_vertex_and_neighbors(origin)?;
                let origin_neighbors = origin_neighbors_guard.get_neighbors();
                let anchor_map = anchor_sets[origin as usize].lock().unwrap();

                for neighbor in origin_neighbors.iter() {
                    // Only include bidirectional neighbors
                    let is_bidirectional = self.graph.contains_edge(*neighbor, origin)?;

                    if is_bidirectional {
                        // Add candidates pruned through this neighbor (anchor)
                        if let Some(candidates) = anchor_map.get(&neighbor) {
                            target_set.extend(candidates.iter().copied());
                        }
                        // Add the neighbor itself
                        target_set.insert(*neighbor);
                    }
                }

                *candidates = target_set;
                Ok::<(), ANNError>(())
            })?;

        Ok(())
    }

    // ─── Build-time search (uses ndarray) ───

    /// Greedy search for candidates during graph construction.
    /// Returns (result_candidates, npq_candidates) where npq_candidates contains
    /// all nodes found in the NeighborPriorityQueue (proximity neighbors for candidate sets).
    fn greedy_search_for_candidates(&self, query_idx: u32) -> ANNResult<Vec<u32>> {
        let mut rng = rand::rng();
        let mut entry = self.entry_point;
        while entry == query_idx {
            entry = rng.random_range(0..self.data.nrows()) as u32;
        }

        let query = self.data.row(query_idx as usize);
        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(self.search_list_size);
        neighbor_pq.insert(Neighbor::new(
            entry,
            l2_distance(query, self.data.row(entry as usize)),
        ));

        let mut result = Vec::new();
        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            result.push(neighbor.id);

            let neighbor_neighbors = self.graph.to_neighbor_vec(neighbor.id)?;

            for &nn in &neighbor_neighbors {
                let dist = l2_distance(query, self.data.row(nn as usize));
                neighbor_pq.insert(Neighbor::new(nn, dist));
            }
        }

        Ok(result)
    }

    // ─── Search: uses SIMD FullPrecisionDistance ───

    pub fn search(&self, query: &[f32; N], k: usize, search_list_size: usize) -> Vec<u32> {
        let entry = self.entry_point;
        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(
            entry,
            <[f32; N]>::distance_compare(query, &self.data_arrays[entry as usize], Metric::L2),
        ));

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            if let Ok(neighbor_neighbors) = self.graph.to_neighbor_vec(neighbor.id) {
                for &nn in &neighbor_neighbors {
                    let dist = <[f32; N]>::distance_compare(
                        query,
                        &self.data_arrays[nn as usize],
                        Metric::L2,
                    );
                    neighbor_pq.insert(Neighbor::new(nn, dist));
                }
            }
        }

        neighbor_pq
            .neighbors()
            .iter()
            .map(|n| n.id)
            .take(k)
            .collect()
    }

    /// Search with ndarray query (for compatibility with build-time callers).
    pub fn search_ndarray(
        &self,
        query: ArrayView1<f32>,
        k: usize,
        search_list_size: usize,
    ) -> Vec<u32> {
        let mut q = [0.0f32; N];
        q.copy_from_slice(query.as_slice().unwrap());
        self.search(&q, k, search_list_size)
    }

    pub fn pq_search(&self, query: &[f32; N], k: usize, search_list_size: usize) -> Vec<u32> {
        let pq = self
            .pq
            .as_ref()
            .expect("Product quantizer is none, please train the pq first.");
        let query_view = ndarray::ArrayView1::from(query.as_slice());
        let tables = pq.compute_adc_table(&query_view);
        let mut visited = HashSet::<u32>::new();

        let entry = self.entry_point;
        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(
            entry,
            pq.adc_distance(entry as usize, &tables),
        ));

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            visited.insert(neighbor.id);
            if let Ok(neighbor_neighbors) = self.graph.to_neighbor_vec(neighbor.id) {
                for &nn in &neighbor_neighbors {
                    let dist = pq.adc_distance(nn as usize, &tables);
                    neighbor_pq.insert(Neighbor::new(nn, dist));
                }
            }
        }

        // Rerank with exact SIMD distance
        let mut result: Vec<(u32, f32)> = visited
            .iter()
            .map(|&id| {
                let dist =
                    <[f32; N]>::distance_compare(query, &self.data_arrays[id as usize], Metric::L2);
                (id, dist)
            })
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        result.iter().map(|(id, _)| *id).take(k).collect()
    }

    // ─── IO ───

    fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let meta = DiskANNOnDisk {
            alpha: self.alpha,
            graph_degree: self.graph_degree,
            search_list_size: self.search_list_size,
            entry_point: self.entry_point,
            graph: self.graph.to_hashmap(),
            candidate_set_manager: self.candidate_set_manager.clone(),
            use_pq: self.use_pq,
            graph_save_path: self.graph_save_path.clone(),
        };

        let mut writer = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut writer, config)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: DiskANNOnDisk = bincode::serde::decode_from_std_read(&mut reader, config)?;

        self.alpha = meta.alpha;
        self.graph_degree = meta.graph_degree;
        self.search_list_size = meta.search_list_size;
        self.entry_point = meta.entry_point;
        self.graph =
            InMemoryGraph::from_hashmap(&meta.graph, self.data.nrows(), self.graph_degree as u32);
        self.candidate_set_manager = meta.candidate_set_manager;
        self.use_pq = meta.use_pq;
        self.graph_save_path = meta.graph_save_path;

        Ok(())
    }
}
