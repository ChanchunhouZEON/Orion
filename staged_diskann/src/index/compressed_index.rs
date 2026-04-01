/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::algorithm::clustering::{CohesiveClusterManager, INVALID_CLUSTER_AFFILIATION};
use crate::model::CompressedGraph;
use crate::model::FixedChunkPQTable;
use crate::model::compressed_graph::CompressedGraphOnDisk;
use crate::model::scratch::InMemScratchPool;
use crate::utils::{DELIMITER_LENGTH, l2_distance};
use diskann::common::{ANNError, ANNResult};
use diskann::model::InMemoryGraph;
use ndarray::{ArcArray1, ArcArray2, Array1};
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
    pub data: ArcArray2<f32>,
    /// Fixed-size arrays for SIMD-accelerated reranking
    pub data_arrays: Vec<[f32; N]>,
    /// Single merged graph with per-node compressed_degree, following diskann's final_graph pattern
    pub graph: CompressedGraph,
    pub candidate_sets: Arc<Vec<HashSet<u32>>>,
    pub entry: u32,
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
    /// Mirrors `CompressedVertexAndNeighbors` but without RwLock or pointer
    /// chasing:
    ///
    /// - `search_csr_offsets[i]`       → start index of node `i` in `search_csr_neighbors`
    /// - `search_csr_offsets[i+1]`     → exclusive end (all neighbors)
    /// - `search_csr_compressed_end[i]`→ exclusive end of compressed portion
    ///
    /// Full neighbors:       `neighbors[offsets[i]..offsets[i+1]]`
    /// Compressed neighbors: `neighbors[offsets[i]..compressed_end[i]]`
    pub(crate) search_csr_offsets: Vec<u32>,
    pub(crate) search_csr_compressed_end: Vec<u32>,
    pub(crate) search_csr_neighbors: Vec<u32>,
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build a CompressedDiskANN: constructs the compressed graph from the InMemoryGraph,
    /// runs clustering (or loads from disk), and returns a fully initialized index.
    ///
    /// Takes ownership of `inmem_graph`; it is consumed during construction and not stored.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: ArcArray2<f32>,
        inmem_graph: InMemoryGraph,
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
        assert_eq!(
            data.ncols(),
            N,
            "Data dimension {} does not match const generic N={}",
            data.ncols(),
            N
        );

        let num_nodes = data.nrows();

        let mut num_pq_chunks: Option<usize> = None;
        if let Some(ref pq) = pq {
            num_pq_chunks = Some(pq.get_num_chunks());
        }

        // Algorithm 7 parameters: m (inter-cluster limit), n (edges per external cluster)
        let max_connection_clusters = max_connection_clusters;
        let max_edges_per_cluster = max_connection_per_cluster;
        // Max degree = m * n external edges
        let max_pruned_degree = max_connection_clusters * max_edges_per_cluster;

        let compressed_graph_save_path = compressed_graph_save_path.unwrap_or_else(|| {
            let dir = PathBuf::from("compressed_dskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!(
                "compressed_dskann_graph_n{}_m{}_p{}_c{}.bin",
                num_nodes, max_cluster_point_size, max_pruned_degree, critical_minimum_rate
            ))
        });

        // Convert ndarray rows to fixed-size arrays for SIMD reranking
        let data_arrays = Self::build_data_arrays(&data);

        // Build CompressedGraph from the InMemoryGraph (imports all neighbors, compressed_degree = 0)
        let compressed_graph = CompressedGraph::from_inmem_graph(&inmem_graph);

        let mut result = StagedDiskANN {
            data: data.clone(),
            data_arrays,
            graph: compressed_graph,
            candidate_sets,
            entry,
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
            search_csr_offsets: Vec::new(),
            search_csr_compressed_end: Vec::new(),
            search_csr_neighbors: Vec::new(),
        };

        // Try to load from disk first; if not found, run clustering
        let _ = result.load_or_build_compressed_graph(inmem_graph);

        // Build flat CSR adjacency list AFTER graph is finalized.
        // Clustering reorders neighbors (compressed-first), so this must come last.
        result.build_search_csr();

        // candidate_sets is only referenced during clustering; free it now.
        result.candidate_sets = Arc::new(vec![]);

        // For non-visualization builds, also release the raw ndarray data and the
        // cluster-metadata fields that are only needed to draw graphs/PCA projections.
        #[cfg(not(feature = "visualization"))]
        {
            result.data = ndarray::Array2::<f32>::zeros((0, 0)).into_shared();
            result.point_affiliation = ndarray::Array1::<i32>::zeros(0).into_shared();
            result.cluster_centroids.clear();
            result.cluster_centroids.shrink_to_fit();
            result.storage_layout.clear();
            result.storage_layout.shrink_to_fit();
        }

        result
    }

    /// Convenience: build from a reference by reconstructing the graph.
    /// Used by call sites that only have `&InMemoryGraph`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_graph_ref(
        data: ArcArray2<f32>,
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
        let graph_map = inmem_graph.to_hashmap();
        let graph_copy =
            InMemoryGraph::from_hashmap(&graph_map, data.nrows(), inmem_graph.max_degree());
        Self::new(
            data,
            graph_copy,
            candidate_sets,
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

        self.search_csr_offsets = offsets;
        self.search_csr_compressed_end = compressed_end;
        self.search_csr_neighbors = neighbors;
    }

    fn build_data_arrays(data: &ArcArray2<f32>) -> Vec<[f32; N]> {
        let n = data.nrows();
        let mut arrays = Vec::with_capacity(n);
        for i in 0..n {
            let row = data.row(i);
            let mut arr = [0.0f32; N];
            arr.copy_from_slice(row.as_slice().unwrap());
            arrays.push(arr);
        }
        arrays
    }

    /// Load from disk or run clustering to build the compressed graph.
    /// The `inmem_graph` reference is only used when clustering is needed (not when loading).
    fn load_or_build_compressed_graph(&mut self, inmem_graph: InMemoryGraph) -> ANNResult<()> {
        if self
            .load(<PathBuf as AsRef<Path>>::as_ref(
                &self.compressed_graph_save_path.clone(),
            ))
            .is_ok()
        {
            log::info!("⌛️: existing compressed graph structure loaded");
            log::info!("{}", "-".repeat(DELIMITER_LENGTH));
            return Ok(());
        }

        log::info!("🏗️: clustering and compressing graph structure");
        let total_start = Instant::now();

        // 1. Run CohesiveClusterManager directly (no trait dispatch)
        let num_nodes = self.data.nrows();
        let graph_arc = Arc::new(inmem_graph);
        let mut manager = CohesiveClusterManager::new(
            num_nodes as u32,
            self.data.clone(),
            graph_arc.clone(),
            self.candidate_sets.clone(),
            self.max_cluster_point_size,
            self.critical_minimum_rate,
        );

        log::info!("🌎: clustering with CohesiveClustering");
        let cluster_start = Instant::now();
        manager
            .construct_cohesive_clusters()
            .map_err(|_| ANNError::log_cluster_error("Cluster failed!".to_string()))?;
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
        drop(graph_arc);
        Ok(())
    }

    /// Build the compressed graph by reordering neighbors in the main graph.
    ///
    /// For each node, Algorithm 7 selects inter-cluster edges. These become
    /// the "compressed" neighbors placed at the front of the neighbor list.
    /// The remaining original neighbors follow. The per-node `compressed_degree`
    /// is stored inside the `CompressedGraph`'s `CompressedVertexAndNeighbors`.
    fn build_compressed_graph(&mut self) {
        use rayon::prelude::*;

        let num_nodes = self.data.nrows();
        let max_connection_clusters = self.max_connection_clusters;
        let max_edges_per_cluster = self.max_edges_per_cluster;

        log::info!(
            "After clustering: {} clusters, m={}, n={}",
            self.storage_layout.len(),
            max_connection_clusters,
            max_edges_per_cluster
        );

        // ── CSR build (used for lock-free neighbor reads in both phases) ──────
        let t_csr = Instant::now();
        self.build_search_csr();
        log::info!("  csr_build:      {:.3}s", t_csr.elapsed().as_secs_f32());

        // ── Phase A: label cross-cluster edges (parallel, no distance calc) ──
        //
        // For each node we scan its CSR neighbors once (O(degree)).
        // Neighbors are grouped by cluster affiliation; we cap at
        // max_connection_clusters distinct external clusters and
        // max_edges_per_cluster members per cluster.
        // Ranking uses the CSR order (i.e. Vamana search-list rank) as a
        // cheap proxy for distance — the first mention of a cluster in the
        // search list is taken as its representative index.
        let t_label = Instant::now();
        let per_node_compressed: Vec<(u32, Vec<u32>)> = {
            let point_affiliation = self.point_affiliation.as_slice().unwrap();
            let csr_offsets = &self.search_csr_offsets;
            let csr_neighbors = &self.search_csr_neighbors;
            let max_ext_clusters = self.max_connection_clusters;
            let max_edges = self.max_edges_per_cluster;
            let max_pruned = self.max_pruned_degree;

            (0..num_nodes as u32)
                .into_par_iter()
                .map(|origin| {
                    let my_cluster = point_affiliation[origin as usize];
                    if my_cluster == INVALID_CLUSTER_AFFILIATION {
                        return (origin, Vec::new());
                    }

                    let start = csr_offsets[origin as usize] as usize;
                    let end = csr_offsets[origin as usize + 1] as usize;
                    let nbrs = &csr_neighbors[start..end];
                    if nbrs.is_empty() {
                        return (origin, Vec::new());
                    }

                    // Accumulate neighbors grouped by external cluster.
                    // Key: cluster affiliation (i32); value: (first_csr_idx, [node, ...]).
                    // Using a small Vec instead of HashMap: degree ≤ 64 → linear scan wins.
                    let mut groups: Vec<(i32, usize, Vec<u32>)> = Vec::new();

                    for (idx, &nn) in nbrs.iter().enumerate() {
                        let aff = point_affiliation[nn as usize];
                        if aff == INVALID_CLUSTER_AFFILIATION || aff == my_cluster {
                            continue;
                        }
                        match groups.iter_mut().find(|g| g.0 == aff) {
                            Some(g) => {
                                if g.2.len() < max_edges {
                                    g.2.push(nn);
                                }
                            }
                            None => {
                                if groups.len() < max_ext_clusters {
                                    groups.push((aff, idx, vec![nn]));
                                }
                            }
                        }
                    }

                    if groups.is_empty() {
                        return (origin, Vec::new());
                    }

                    // Sort groups by the CSR index of their first (nearest) member.
                    groups.sort_unstable_by_key(|g| g.1);

                    let mut compressed = Vec::with_capacity(max_pruned);
                    for (_, _, members) in &groups {
                        compressed.extend_from_slice(members);
                    }

                    (origin, compressed)
                })
                .collect()
        };
        log::info!("  label_edges:    {:.3}s", t_label.elapsed().as_secs_f32());

        // ── Phase A reorder: move compressed-labeled edges to front (parallel) ─
        //
        // All compressed_nbrs are already present in the CSR (they came from it),
        // so no new edges are added — we only reorder.
        //
        // Reads use the already-built CSR (lock-free).
        // Writes use per-node RwLock inside CompressedGraph (&self interior mut).
        // Different threads write to different node slots → no contention.
        let t_reorder = Instant::now();
        {
            let graph = &self.graph;
            let csr_offsets = &self.search_csr_offsets;
            let csr_neighbors = &self.search_csr_neighbors;

            per_node_compressed
                .par_iter()
                .for_each(|(origin, compressed_nbrs)| {
                    if compressed_nbrs.is_empty() {
                        return;
                    }

                    // Lock-free read from CSR.
                    let start = csr_offsets[*origin as usize] as usize;
                    let end = csr_offsets[*origin as usize + 1] as usize;
                    let all_nbrs = &csr_neighbors[start..end];

                    // Partition: compressed first, rest after.
                    // compressed_nbrs is small (≤ m×n ≈ 12) → linear-scan.
                    let mut compressed = Vec::with_capacity(compressed_nbrs.len());
                    let mut rest = Vec::with_capacity(all_nbrs.len());
                    for &n in all_nbrs {
                        if compressed_nbrs.contains(&n) {
                            compressed.push(n);
                        } else {
                            rest.push(n);
                        }
                    }

                    // Per-node write lock — different nodes in different threads.
                    graph
                        .set_neighbors_split(*origin, &compressed, &rest)
                        .unwrap();
                });
        }
        log::info!("  reorder:        {:.3}s", t_reorder.elapsed().as_secs_f32());

        let t_maxcd = Instant::now();
        self.graph.update_max_compressed_degree();
        log::info!("  update_max_cd:  {:.3}s", t_maxcd.elapsed().as_secs_f32());

        // ── Phase B: connectivity repair ──────────────────────────────────────
        // Vamana graphs are always strongly connected; this is a zero-cost
        // safety check in practice.
        let t_repair = Instant::now();
        self.repair_connectivity();
        log::info!("  repair:         {:.3}s", t_repair.elapsed().as_secs_f32());
    }

    /// BFS from entry point to find unreachable nodes using ALL neighbors, then connect them.
    fn repair_connectivity(&mut self) {
        let num_nodes = self.data.nrows();
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
                    let dist = l2_distance(self.data.row(u as usize), self.data.row(n as usize));
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

        log::info!("Computing PCA reduction to 2D...");
        log::info!("PCA reduction complete: {} points -> 2D", self.data.nrows());

        let graph_map = self.graph.to_hashmap();
        let compressed_map = self.graph.to_compressed_hashmap();

        let path = format!("{}/original_graph.png", output_dir);
        log::info!("Drawing original graph -> {}", path);
        visualization::draw_graph(&self.data, &graph_map, &path)?;

        let path = format!("{}/clustered_graph.png", output_dir);
        log::info!("Drawing clustered graph -> {}", path);
        visualization::draw_clustered_graph(
            &self.data,
            self.point_affiliation.as_slice().unwrap(),
            &self.storage_layout,
            &graph_map,
            &path,
        )?;

        let path = format!("{}/compressed_graph.png", output_dir);
        log::info!("Drawing compressed graph -> {}", path);
        visualization::draw_compressed_graph(
            &self.data,
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
                    &self.data, cluster_id, members, centroid, &graph_map, &path,
                )?;
            }
        }
        log::info!(
            "Drew {} individual cluster visualizations",
            cluster_ids.len().min(20)
        );

        let metrics = visualization::compute_clustering_metrics(
            &self.data,
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
