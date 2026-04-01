/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::algorithm::clustering::{CohesiveClusterManager, INVALID_CLUSTER_AFFILIATION};
use crate::model::CompressedGraph;
use crate::model::FixedChunkPQTable;
use crate::model::compressed_graph::CompressedGraphOnDisk;
use crate::model::scratch::InMemScratchPool;
use crate::utils::DELIMITER_LENGTH;
use diskann::common::{ANNError, ANNResult};
use diskann::model::{CsrGraph, InMemoryGraph, InmemDataset};
use ndarray::{ArcArray1, Array1};
#[cfg(feature = "visualization")]
use ndarray::ArcArray2;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use vector::FullPrecisionDistance;

/// Compressed DiskANN with cluster-aware graph pruning and two-phase search.
/// Const-generic `N` enables SIMD-accelerated distance computation during search reranking.
///
/// Uses a single `CompressedGraph` following diskann-core's `final_graph` pattern
/// (`Vec<RwLock<CompressedVertexAndNeighbors>>`). Each node's neighbors are ordered:
/// - First `compressed_degree` neighbors = "compressed" (used in phase 2)
/// - Remaining neighbors = "full" (used only in phase 1)
pub struct StagedDiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Vector data storage taken from the DiskANN `InmemIndex`.
    /// All distance computations go through `dataset.get_vertex()` +
    /// `Vertex::compare()` — the exact same code path as DiskANN's search.
    pub dataset: InmemDataset<f32, N>,
    /// Single merged graph with per-node compressed_degree, following diskann's final_graph pattern
    pub graph: CompressedGraph,
    pub candidate_sets: Arc<Vec<HashSet<u32>>>,
    pub entry: u32,
    pub num_nodes: usize,
    pub compressed_graph_save_path: PathBuf,
    pub storage_layout: HashMap<u32, HashSet<u32>>,
    pub point_affiliation: ArcArray1<i32>,
    /// Per-cluster centroid point IDs from clustering (used in Algorithm 7).
    cluster_centroids: HashMap<u32, u32>,

    // Pruning parameters
    pub max_connection_clusters: usize,
    pub max_edges_per_cluster: usize,
    pub max_pruned_degree: usize,
    pub max_cluster_point_size: usize,
    pub critical_minimum_rate: f32,

    // Saving the result or not
    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,

    /// Pool of pre-allocated scratch spaces for in-memory search.
    /// Lazily initialized on first call to `search()`.
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,

    /// Flat CSR adjacency list for lock-free in-memory search.
    ///
    /// `search_csr_offsets` and `search_csr_neighbors` are `Arc<Vec<u32>>`
    /// shared with the input `CsrGraph` — no data copy on construction.
    /// After label+reorder, they are replaced with newly built arcs.
    ///
    /// - `search_csr_offsets[i]`       → start index of node `i` in `search_csr_neighbors`
    /// - `search_csr_offsets[i+1]`     → exclusive end (all neighbors)
    /// - `search_csr_compressed_end[i]`→ exclusive end of compressed portion
    ///
    /// Full neighbors:       `neighbors[offsets[i]..offsets[i+1]]`
    /// Compressed neighbors: `neighbors[offsets[i]..compressed_end[i]]`
    pub(crate) search_csr_offsets: Arc<Vec<u32>>,
    pub(crate) search_csr_compressed_end: Vec<u32>,
    pub(crate) search_csr_neighbors: Arc<Vec<u32>>,
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build a StagedDiskANN from a lock-free `CsrGraph` and an `InmemDataset`.
    ///
    /// `dataset` is taken from the DiskANN `InmemIndex` (via downcast + `mem::take`).
    /// All distance computations use `dataset.get_vertex()` + `Vertex::compare()`,
    /// the exact same code path as DiskANN's greedy search.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: InmemDataset<f32, N>,
        csr: CsrGraph,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        bidir_neighbors: Arc<Vec<Vec<u32>>>,
        entry: u32,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
        max_cluster_point_size: usize,
        max_connection_clusters: usize,
        max_connection_per_cluster: usize,
        critical_minimum_rate: f32,
        compressed_graph_save_path: Option<PathBuf>,
        is_save: bool,
    ) -> Self {
        let num_nodes = csr.num_nodes();

        let mut num_pq_chunks: Option<usize> = None;
        if let Some(ref pq) = pq {
            num_pq_chunks = Some(pq.get_num_chunks());
        }

        let max_edges_per_cluster = max_connection_per_cluster;
        let max_pruned_degree = max_connection_clusters * max_edges_per_cluster;

        let compressed_graph_save_path = compressed_graph_save_path.unwrap_or_else(|| {
            let dir = PathBuf::from("compressed_dskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!(
                "compressed_dskann_graph_n{}_m{}_p{}_c{}.bin",
                num_nodes, max_cluster_point_size, max_pruned_degree, critical_minimum_rate
            ))
        });

        // Build CompressedGraph from CSR (needed for repair / save).
        let t_cg = Instant::now();
        let compressed_graph = CompressedGraph::from_csr(&csr, csr.max_degree);
        log::info!("  from_csr:           {:.3}s", t_cg.elapsed().as_secs_f32());

        let init_compressed_end = vec![0u32; num_nodes];

        let mut result = StagedDiskANN {
            dataset,
            graph: compressed_graph,
            candidate_sets,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            compressed_graph_save_path,
            storage_layout: HashMap::new(),
            point_affiliation: Array1::from_elem(num_nodes, -1).to_shared(),
            cluster_centroids: HashMap::new(),
            max_connection_clusters,
            max_edges_per_cluster,
            max_pruned_degree,
            max_cluster_point_size,
            critical_minimum_rate,
            is_save,
            inmem_scratch_pool: OnceLock::new(),
            search_csr_offsets: csr.offsets,
            search_csr_compressed_end: init_compressed_end,
            search_csr_neighbors: csr.neighbors,
        };

        // Run clustering + label + reorder (writes final CSR in one pass).
        let _ = result.load_or_build_compressed_graph(bidir_neighbors);

        result.candidate_sets = Arc::new(vec![]);

        #[cfg(not(feature = "visualization"))]
        {
            result.point_affiliation = ndarray::Array1::<i32>::zeros(0).into_shared();
            result.cluster_centroids.clear();
            result.cluster_centroids.shrink_to_fit();
            result.storage_layout.clear();
            result.storage_layout.shrink_to_fit();
        }

        result
    }

    /// Convenience: build from a `&InMemoryGraph` reference.
    #[allow(clippy::too_many_arguments)]
    pub fn from_graph_ref(
        dataset: InmemDataset<f32, N>,
        inmem_graph: &InMemoryGraph,
        candidate_sets: Arc<Vec<HashSet<u32>>>,
        entry: u32,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
        max_cluster_point_size: usize,
        max_connection_clusters: usize,
        max_connection_per_cluster: usize,
        critical_minimum_rate: f32,
        compressed_graph_save_path: Option<PathBuf>,
        is_save: bool,
    ) -> Self {
        let t_copy = Instant::now();
        let n = inmem_graph.size();
        let adj: Vec<Vec<u32>> = (0..n as u32)
            .map(|i| inmem_graph.to_neighbor_vec(i).unwrap_or_default())
            .collect();
        let csr = CsrGraph::from_adjacency_list(adj);
        log::info!("  graph_to_csr:   {:.3}s", t_copy.elapsed().as_secs_f32());
        Self::new(
            dataset,
            csr,
            candidate_sets,
            Arc::new(Vec::new()), // bidir not pre-computed; clustering will compute it
            entry,
            pq,
            pq_codes,
            max_cluster_point_size,
            max_connection_clusters,
            max_connection_per_cluster,
            critical_minimum_rate,
            compressed_graph_save_path,
            is_save,
        )
    }

    /// Convert the dataset to `ArcArray2<f32>` for visualization.
    #[cfg(feature = "visualization")]
    pub fn data_as_arc_array2(&self) -> ArcArray2<f32> {
        let n = self.num_nodes;
        let flat: Vec<f32> = self.dataset.get_data()[..n * N].to_vec();
        ndarray::Array2::from_shape_vec((n, N), flat)
            .expect("shape mismatch")
            .into_shared()
    }

    /// Return per-node (full_degree, compressed_degree) from the CSR.
    /// Used for diagnostics.
    pub fn csr_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.search_csr_offsets.len().saturating_sub(1);
        (0..n)
            .map(|i| {
                let start = self.search_csr_offsets[i] as usize;
                let end = self.search_csr_offsets[i + 1] as usize;
                let cd_end = self.search_csr_compressed_end[i] as usize;
                (end - start, cd_end - start)
            })
            .collect()
    }

    /// Build flat CSR adjacency list from the finalized compressed graph.
    ///
    /// Acquires each RwLock exactly once at construction time; search then
    /// uses direct slice indexing with no locking or pointer chasing.
    fn build_search_csr(&mut self) {
        let n = self.graph.size();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut compressed_end = Vec::with_capacity(n);
        let mut neighbors: Vec<u32> = Vec::new();

        offsets.push(0u32);
        for i in 0..n as u32 {
            if let Ok(v) = self.graph.read_vertex(i) {
                let start = neighbors.len() as u32;
                neighbors.extend_from_slice(v.get_neighbors());
                let cd = v.compressed_degree().min(v.degree() as u32);
                compressed_end.push(start + cd);
            } else {
                // Degenerate: node unreadable — record zero-length entry.
                compressed_end.push(*offsets.last().unwrap());
            }
            offsets.push(neighbors.len() as u32);
        }

        self.search_csr_offsets = Arc::new(offsets);
        self.search_csr_compressed_end = compressed_end;
        self.search_csr_neighbors = Arc::new(neighbors);
    }


    /// Load from disk or run clustering to build the compressed graph.
    ///
    /// When building fresh, constructs an `Arc<CsrGraph>` from the CSR stored in
    /// `self.search_csr_*` for the clustering phase — no `InMemoryGraph` needed.
    fn load_or_build_compressed_graph(&mut self, bidir_neighbors: Arc<Vec<Vec<u32>>>) -> ANNResult<()> {
        if self
            .load(<PathBuf as AsRef<Path>>::as_ref(
                &self.compressed_graph_save_path.clone(),
            ))
            .is_ok()
        {
            log::info!("⌛️: existing compressed graph structure loaded");
            // Rebuild search CSR from the loaded CompressedGraph.
            self.build_search_csr();
            log::info!("{}", "-".repeat(DELIMITER_LENGTH));
            return Ok(());
        }

        log::info!("🏗️: clustering and compressing graph structure");
        let total_start = Instant::now();

        // 1. Share the initial search CSR for clustering (Arc::clone = O(1)).
        let num_nodes = self.num_nodes;
        let csr_for_cluster = Arc::new(CsrGraph {
            offsets: self.search_csr_offsets.clone(),   // Arc::clone, O(1)
            neighbors: self.search_csr_neighbors.clone(), // Arc::clone, O(1)
            max_degree: self.graph.max_degree(),
        });
        let mut manager = CohesiveClusterManager::new(
            num_nodes as u32,
            &self.dataset,
            csr_for_cluster,
            self.candidate_sets.clone(),
            self.max_cluster_point_size,
            self.critical_minimum_rate,
        );

        log::info!("🌎: clustering with CohesiveClustering");
        let cluster_start = Instant::now();
        if bidir_neighbors.is_empty() {
            manager
                .construct_cohesive_clusters()
                .map_err(|_| ANNError::log_cluster_error("Cluster failed!".to_string()))?;
        } else {
            manager
                .construct_cohesive_clusters_with_bidir(bidir_neighbors)
                .map_err(|_| ANNError::log_cluster_error("Cluster failed!".to_string()))?;
        }
        self.point_affiliation = manager.point_affiliation.clone();
        let clusters = manager.cohesive_clusters.into_inner();
        let cluster_time = cluster_start.elapsed();
        log::info!(
            "🌎: clustering time (CohesiveClustering): {:.2}s",
            cluster_time.as_secs_f32()
        );

        // 2. Build compressed graph (reorder neighbors in-place)
        log::info!("✂️: building compressed graph based on clusters.");
        let prune_start = Instant::now();
        self.build_compressed_graph();
        let prune_time = prune_start.elapsed();
        log::info!("✂️: pruning time: {:.2}s", prune_time.as_secs_f32());

        log::info!(
            "⌛️: total CompressedDiskANN overhead: {:.2}s (clustering: {:.2}s, pruning: {:.2}s)",
            total_start.elapsed().as_secs_f32(),
            cluster_time.as_secs_f32(),
            prune_time.as_secs_f32()
        );

        // 3. Save the compressed graph
        if self.is_save {
            if let Err(e) = self.save(&self.compressed_graph_save_path.clone()) {
                log::warn!("Failed to save compressed graph: {}", e);
            }
        }

        log::info!("{}", "-".repeat(DELIMITER_LENGTH));

        self.cluster_centroids = clusters
            .iter()
            .filter_map(|(&k, v)| v.centroid().map(|c| (k, c)))
            .collect();
        self.storage_layout = clusters
            .iter()
            .map(|(&k, v)| (k, v.cluster_point().keys().copied().collect()))
            .collect();
        drop(clusters);
        Ok(())
    }

    /// Build the compressed graph: fused label + reorder in a single parallel pass.
    ///
    /// For each node, neighbors are grouped by cluster affiliation (all clusters
    /// are treated uniformly — no internal/external distinction). The top
    /// `max_connection_clusters` clusters are selected, each contributing up to
    /// `max_edges_per_cluster` neighbors. These form the "compressed" portion,
    /// placed at the front of the reordered neighbor list. The remaining
    /// neighbors follow.
    ///
    /// The result is written directly as a new search CSR with per-node
    /// `compressed_end`, then synced back to `CompressedGraph` for save/repair.
    fn build_compressed_graph(&mut self) {
        use rayon::prelude::*;

        let num_nodes = self.num_nodes;

        log::info!(
            "  m={}, n={}, max_pruned_degree={}",
            self.max_connection_clusters,
            self.max_edges_per_cluster,
            self.max_pruned_degree,
        );

        // ── Fused label + reorder (parallel) ─────────────────────────────────
        //
        // Each node scans its CSR neighbors once. Neighbors are grouped by
        // cluster; the top-m clusters (by first-appearance rank in the Vamana
        // search list) contribute up to n edges each. The neighbor list is
        // then partitioned into [compressed | rest] in one pass.
        let t_fused = Instant::now();

        let point_affiliation = self.point_affiliation.as_slice().unwrap();
        let csr_offsets = &self.search_csr_offsets;
        let csr_neighbors = &self.search_csr_neighbors;
        let max_clusters = self.max_connection_clusters;
        let max_edges = self.max_edges_per_cluster;

        // Parallel: each node → (compressed_degree, reordered_neighbors)
        let per_node: Vec<(u32, Vec<u32>)> = (0..num_nodes)
            .into_par_iter()
            .map(|origin| {
                let start = csr_offsets[origin] as usize;
                let end = csr_offsets[origin + 1] as usize;
                let nbrs = &csr_neighbors[start..end];

                if point_affiliation[origin] == INVALID_CLUSTER_AFFILIATION || nbrs.is_empty() {
                    return (0u32, nbrs.to_vec());
                }

                // Group neighbors by cluster affiliation.
                // (cluster_id, first_csr_idx, member_indices_bitset)
                // We track member indices within `nbrs` as bits in a u64,
                // avoiding Vec allocations and enabling O(1) membership test.
                // degree ≤ 64 so a single u64 suffices.
                let mut groups: Vec<(i32, usize, u64, u32)> = Vec::new(); // (aff, first_idx, member_bits, count)

                for (idx, &nn) in nbrs.iter().enumerate() {
                    let aff = point_affiliation[nn as usize];
                    if aff == INVALID_CLUSTER_AFFILIATION {
                        continue;
                    }
                    match groups.iter_mut().find(|g| g.0 == aff) {
                        Some(g) => {
                            if (g.3 as usize) < max_edges {
                                g.2 |= 1u64 << idx;
                                g.3 += 1;
                            }
                        }
                        None => {
                            if groups.len() < max_clusters {
                                groups.push((aff, idx, 1u64 << idx, 1));
                            }
                        }
                    }
                }

                if groups.is_empty() {
                    return (0u32, nbrs.to_vec());
                }

                // Rank clusters by first-appearance index in the search list.
                groups.sort_unstable_by_key(|g| g.1);

                // Merge all member bits into a single compressed bitset.
                let mut compressed_bits: u64 = 0;
                for &(_, _, bits, _) in &groups {
                    compressed_bits |= bits;
                }
                let cd = compressed_bits.count_ones();

                // Partition: compressed first (in group order), rest after.
                let mut reordered = Vec::with_capacity(nbrs.len());
                for &(_, _, bits, _) in &groups {
                    let mut b = bits;
                    while b != 0 {
                        let idx = b.trailing_zeros() as usize;
                        reordered.push(nbrs[idx]);
                        b &= b - 1; // clear lowest set bit
                    }
                }
                for (idx, &n) in nbrs.iter().enumerate() {
                    if compressed_bits & (1u64 << idx) == 0 {
                        reordered.push(n);
                    }
                }

                (cd, reordered)
            })
            .collect();

        // ── Pack into flat CSR ───────────────────────────────────────────────
        let mut new_offsets = Vec::with_capacity(num_nodes + 1);
        let mut new_neighbors: Vec<u32> = Vec::new();
        let mut new_compressed_end = Vec::with_capacity(num_nodes);

        new_offsets.push(0u32);
        for (cd, nbrs) in &per_node {
            let off = new_neighbors.len() as u32;
            new_neighbors.extend_from_slice(nbrs);
            new_compressed_end.push(off + cd);
            new_offsets.push(new_neighbors.len() as u32);
        }

        self.search_csr_offsets = Arc::new(new_offsets);
        self.search_csr_neighbors = Arc::new(new_neighbors);
        self.search_csr_compressed_end = new_compressed_end;

        log::info!("  label+reorder:  {:.3}s", t_fused.elapsed().as_secs_f32());

        // ── Sync CompressedGraph from new CSR (needed for save / repair) ─────
        let t_sync = Instant::now();
        {
            let graph = &self.graph;
            per_node
                .par_iter()
                .enumerate()
                .for_each(|(i, (cd, nbrs))| {
                    graph
                        .set_neighbors_split(
                            i as u32,
                            &nbrs[..*cd as usize],
                            &nbrs[*cd as usize..],
                        )
                        .unwrap();
                });
        }
        self.graph.update_max_compressed_degree();
        log::info!("  sync_graph:     {:.3}s", t_sync.elapsed().as_secs_f32());

        // ── Connectivity repair ──────────────────────────────────────────────
        // Vamana graphs are strongly connected and label+reorder only reorders
        // neighbors (no edges added or removed), so repair is a no-op in
        // practice. Run only in debug builds to catch regressions.
        #[cfg(debug_assertions)]
        {
            let t_repair = Instant::now();
            self.repair_connectivity();
            log::info!("  repair:         {:.3}s", t_repair.elapsed().as_secs_f32());
        }
        #[cfg(not(debug_assertions))]
        log::info!("  repair:         skipped (release)");
    }

    /// BFS from entry point to find unreachable nodes using ALL neighbors, then connect them.
    fn repair_connectivity(&mut self) {
        let num_nodes = self.num_nodes;
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();

        reachable.insert(self.entry);
        queue.push_back(self.entry);

        while let Some(cur) = queue.pop_front() {
            if let Ok(v) = self.graph.read_vertex(cur) {
                for &n in v.get_neighbors() {
                    if reachable.insert(n) {
                        queue.push_back(n);
                    }
                }
            }
        }

        if reachable.len() == num_nodes {
            log::info!("All {} nodes reachable from entry point.", num_nodes);
            return;
        }

        let unreachable_count = num_nodes - reachable.len();
        log::info!(
            "{} unreachable nodes, attempting repair...",
            unreachable_count
        );

        let mut fixed = 0;
        for u in 0..num_nodes as u32 {
            if reachable.contains(&u) {
                continue;
            }

            // Try graph neighbors first to find a reachable one
            let neighbors = self.graph.to_neighbor_vec(u).unwrap_or_default();
            let mut best_reachable: Option<u32> = None;
            let mut best_dist = f32::MAX;

            for &n in &neighbors {
                if reachable.contains(&n) {
                    let dist = self.dataset.get_distance(u, n, vector::Metric::L2).unwrap_or(f32::MAX);
                    if dist < best_dist {
                        best_dist = dist;
                        best_reachable = Some(n);
                    }
                }
            }

            if let Some(n) = best_reachable {
                // Add edge u -> n (append to rest portion)
                let mut u_nbrs = self.graph.to_neighbor_vec(u).unwrap_or_default();
                let u_cd = self.graph.compressed_degree(u);
                if !u_nbrs.contains(&n) {
                    u_nbrs.push(n);
                    self.graph.set_neighbors(u, u_nbrs, u_cd).unwrap();
                }
                // Add reverse edge n -> u
                let mut n_nbrs = self.graph.to_neighbor_vec(n).unwrap_or_default();
                let n_cd = self.graph.compressed_degree(n);
                if !n_nbrs.contains(&u) {
                    n_nbrs.push(u);
                    self.graph.set_neighbors(n, n_nbrs, n_cd).unwrap();
                }
                reachable.insert(u);
                fixed += 1;
            }
        }

        log::info!(
            "Fixed {} / {} unreachable nodes via graph neighbors.",
            fixed,
            unreachable_count
        );
    }

    // --- Visualization ---

    #[cfg(feature = "visualization")]
    pub fn generate_visualizations(&self, output_dir: &str) -> anyhow::Result<()> {
        use crate::visualization;

        std::fs::create_dir_all(output_dir)?;
        log::info!("Generating visualizations in {}", output_dir);

        let data = self.data_as_arc_array2();
        log::info!("PCA reduction complete: {} points -> 2D", data.nrows());

        let graph_map = self.graph.to_hashmap();
        let compressed_map = self.graph.to_compressed_hashmap();

        let path = format!("{}/original_graph.png", output_dir);
        log::info!("Drawing original graph -> {}", path);
        visualization::draw_graph(&data, &graph_map, &path)?;

        let path = format!("{}/clustered_graph.png", output_dir);
        log::info!("Drawing clustered graph -> {}", path);
        visualization::draw_clustered_graph(
            &data,
            self.point_affiliation.as_slice().unwrap(),
            &self.storage_layout,
            &graph_map,
            &path,
        )?;

        let path = format!("{}/compressed_graph.png", output_dir);
        log::info!("Drawing compressed graph -> {}", path);
        visualization::draw_compressed_graph(
            &data,
            &self.storage_layout,
            &compressed_map,
            &path,
        )?;

        let clusters_dir = format!("{}/clusters", output_dir);
        std::fs::create_dir_all(&clusters_dir)?;

        let mut cluster_ids: Vec<u32> = self.storage_layout.keys().copied().collect();
        cluster_ids.sort();

        for &cluster_id in cluster_ids.iter().take(20) {
            if let Some(members) = self.storage_layout.get(&cluster_id) {
                let centroid: Option<u32> = self.cluster_centroids.get(&cluster_id).copied();
                let path = format!("{}/cluster_{}.png", clusters_dir, cluster_id);
                visualization::draw_single_cluster(
                    &data, cluster_id, members, centroid, &graph_map, &path,
                )?;
            }
        }
        log::info!(
            "Drew {} individual cluster visualizations",
            cluster_ids.len().min(20)
        );

        let metrics = visualization::compute_clustering_metrics(
            &data,
            self.point_affiliation.as_slice().unwrap(),
            &self.storage_layout,
            &graph_map,
            &compressed_map,
        );

        println!("{}", metrics);

        let metrics_path = format!("{}/metrics.txt", output_dir);
        std::fs::write(&metrics_path, format!("{}", metrics))?;
        log::info!("Saved metrics to {}", metrics_path);

        Ok(())
    }

    // --- IO ---

    fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        // Save the graph in its native format
        let graph_on_disk = self.graph.to_on_disk();

        // Wrap with CompressedDiskANN metadata
        let meta = CompressedDiskANNMeta {
            version: 3,
            entry: self.entry,
            storage_layout: self.storage_layout.clone(),
            point_affiliation: self.point_affiliation.to_vec(),
            max_pruned_degree: self.max_pruned_degree,
            max_connection_clusters: self.max_connection_clusters,
        };

        let mut writer = BufWriter::new(File::create(path)?);

        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();

        bincode::serde::encode_into_std_write((&meta, &graph_on_disk), &mut writer, config)?;
        writer.flush()?;

        Ok(())
    }

    fn load<P: AsRef<Path> + ?Sized>(&mut self, path: &P) -> anyhow::Result<()> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let (meta, graph_on_disk): (CompressedDiskANNMeta, CompressedGraphOnDisk) =
            bincode::serde::decode_from_std_read(&mut reader, config)?;

        self.entry = meta.entry;
        self.max_pruned_degree = meta.max_pruned_degree;
        self.max_connection_clusters = meta.max_connection_clusters;

        // Reconstruct the CompressedGraph from on-disk format
        self.graph = CompressedGraph::from_on_disk(graph_on_disk);
        self.storage_layout = meta.storage_layout;
        self.point_affiliation = Array1::from_vec(meta.point_affiliation).to_shared();

        Ok(())
    }
}

/// Serializable metadata for CompressedDiskANN (separate from graph data).
#[derive(serde::Serialize, serde::Deserialize)]
struct CompressedDiskANNMeta {
    version: u32,
    entry: u32,
    storage_layout: HashMap<u32, HashSet<u32>>,
    point_affiliation: Vec<i32>,
    max_pruned_degree: usize,
    max_connection_clusters: usize,
}
