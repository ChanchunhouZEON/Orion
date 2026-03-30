/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::algorithm::clustering::CohesiveClusterManager;
use crate::algorithm::clustering_trait::{ClusteringMethod, ClusteringStrategy};
use crate::algorithm::lpa::LabelPropagationClustering;
use crate::model::CompressedGraph;
use crate::model::FixedChunkPQTable;
use crate::model::compressed_graph::CompressedGraphOnDisk;
use crate::utils::{DELIMITER_LENGTH, l2_distance, l2_distance_slice};
use diskann::common::{ANNError, ANNResult};
use diskann::model::InMemoryGraph;
use ndarray::{ArcArray1, ArcArray2, Array1};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
    pub max_external_clusters: usize,
    pub max_edges_per_ext_cluster: usize,
    pub max_pruned_degree: usize,
    pub max_cluster_point_size: usize,
    pub critical_minimum_rate: f32,

    // Saving the result or not
    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,
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
        clustering_method: ClusteringMethod,
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
        let max_external_clusters = max_connection_clusters;
        let max_edges_per_ext_cluster = max_connection_per_cluster;
        // Max degree = m * n external edges
        let max_pruned_degree = max_external_clusters * max_edges_per_ext_cluster;

        let compressed_graph_save_path = compressed_graph_save_path.unwrap_or_else(|| {
            let dir = PathBuf::from("compressed_dskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!(
                "compressed_dskann_graph_m{}_p{}_c{}.bin",
                max_cluster_point_size, max_pruned_degree, critical_minimum_rate
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
            max_external_clusters,
            max_edges_per_ext_cluster,
            max_pruned_degree,
            max_cluster_point_size,
            critical_minimum_rate,
            is_save,
        };

        // Try to load from disk first; if not found, run clustering
        let _ = result.load_or_build_compressed_graph(inmem_graph, clustering_method);

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
        clustering_method: ClusteringMethod,
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
            clustering_method,
            is_save,
        )
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
    fn load_or_build_compressed_graph(
        &mut self,
        inmem_graph: InMemoryGraph,
        clustering_method: ClusteringMethod,
    ) -> ANNResult<()> {
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

        // 1. Create and run clustering strategy
        let num_nodes = self.data.nrows();
        let graph_arc = Arc::new(inmem_graph);
        let mut clustering: Box<dyn ClusteringStrategy + Send> = match clustering_method {
            ClusteringMethod::Cohesive => Box::new(CohesiveClusterManager::new(
                num_nodes as u32,
                self.data.clone(),
                graph_arc.clone(),
                self.candidate_sets.clone(),
                self.max_cluster_point_size,
                self.critical_minimum_rate,
            )),
            ClusteringMethod::LabelPropagation => Box::new(LabelPropagationClustering::new(
                graph_arc.clone(),
                num_nodes,
                10, // max_iterations
                self.max_cluster_point_size,
            )),
        };

        let strategy_name = clustering.name().to_string();
        log::info!("🌎: clustering with strategy: {}", strategy_name);
        let cluster_start = Instant::now();
        let cluster_result = clustering
            .cluster()
            .map_err(|_| ANNError::log_cluster_error("Cluster failed!".to_string()))?;
        drop(clustering); // Release Arc refs
        drop(graph_arc); // Free InMemoryGraph
        self.point_affiliation = cluster_result.point_affiliation;
        self.storage_layout = cluster_result.storage_layout;
        self.cluster_centroids = cluster_result.centroids;
        let cluster_time = cluster_start.elapsed();
        log::info!(
            "🌎: clustering time ({}): {:.2}s",
            strategy_name,
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
        Ok(())
    }

    // --- Algorithm 7: Cluster-aware Graph Compression ---

    /// Get centroid vectors for each cluster.
    fn compute_cluster_centroids(&self) -> HashMap<u32, Vec<f32>> {
        let dimension = self.data.ncols();
        let mut centroids = HashMap::new();

        for (&cluster_id, members) in &self.storage_layout {
            if members.is_empty() {
                continue;
            }

            // Use stored centroid point if available
            if let Some(&centroid_pid) = self.cluster_centroids.get(&cluster_id) {
                centroids.insert(cluster_id, self.data.row(centroid_pid as usize).to_vec());
                continue;
            }

            // Fallback: compute mean vector
            let mut centroid = vec![0.0f32; dimension];
            for &pid in members {
                let row = self.data.row(pid as usize);
                for (c, &v) in centroid.iter_mut().zip(row.iter()) {
                    *c += v;
                }
            }
            let n = members.len() as f32;
            for c in &mut centroid {
                *c /= n;
            }
            centroids.insert(cluster_id, centroid);
        }

        centroids
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
        let max_external_clusters = self.max_external_clusters;
        let max_edges_per_ext_cluster = self.max_edges_per_ext_cluster;

        // 1. Compute cluster centroids
        let cluster_centroids = self.compute_cluster_centroids();
        log::info!(
            "Algorithm 7: {} clusters, m={}, n={}",
            cluster_centroids.len(),
            max_external_clusters,
            max_edges_per_ext_cluster
        );

        // 2. Phase A (parallel): Per-node compressed edge selection per Algorithm 7
        log::info!("Phase A: Computing compressed edges (Algorithm 7, parallel)...");
        let per_node_compressed: Vec<(u32, Vec<u32>)> = {
            let data = &self.data;
            let point_affiliation = self.point_affiliation.as_slice().unwrap();
            let storage_layout = &self.storage_layout;

            (0..num_nodes as u32)
                .into_par_iter()
                .map(|origin| {
                    let my_cluster = point_affiliation[origin as usize];
                    if my_cluster < 0 {
                        return (origin, Vec::new());
                    }
                    let my_cluster_id = my_cluster as u32;

                    let mut compressed_neighbors = Vec::new();

                    // Step 1: Compute distances to all cluster centroids (excluding own)
                    let origin_row = data.row(origin as usize);
                    let origin_slice = origin_row.as_slice().unwrap();
                    let mut cluster_dists: Vec<(f32, u32)> = cluster_centroids
                        .iter()
                        .filter(|(cid, _)| **cid != my_cluster_id)
                        .map(|(&cid, centroid)| {
                            let dist = l2_distance_slice(origin_slice, centroid);
                            (dist, cid)
                        })
                        .collect();
                    cluster_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    // Step 2: Select m nearest external clusters, add up to n nearest nodes each
                    for &(_, ext_cluster_id) in cluster_dists.iter().take(max_external_clusters) {
                        if let Some(cluster_members) = storage_layout.get(&ext_cluster_id) {
                            let mut member_dists: Vec<(f32, u32)> = cluster_members
                                .iter()
                                .map(|&m| {
                                    let dist = l2_distance(origin_row, data.row(m as usize));
                                    (dist, m)
                                })
                                .collect();
                            member_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                            for &(_, member) in member_dists.iter().take(max_edges_per_ext_cluster)
                            {
                                compressed_neighbors.push(member);
                            }
                        }
                    }

                    (origin, compressed_neighbors)
                })
                .collect()
        };

        // 3. Reorder graph neighbors: compressed first, then rest
        log::info!("Phase A: Reordering graph neighbors (compressed first)...");
        for (origin, compressed_nbrs) in &per_node_compressed {
            let all_nbrs = self.graph.to_neighbor_vec(*origin).unwrap_or_default();
            let compressed_set: HashSet<u32> = compressed_nbrs.iter().copied().collect();

            // Build compressed portion: graph neighbors that are in the compressed set first
            let mut compressed = Vec::new();
            for &n in &all_nbrs {
                if compressed_set.contains(&n) {
                    compressed.push(n);
                }
            }
            // Also add compressed neighbors not already in the graph
            let compressed_already: HashSet<u32> = compressed.iter().copied().collect();
            for &n in compressed_nbrs {
                if !compressed_already.contains(&n) {
                    compressed.push(n);
                }
            }

            // Build rest portion: original neighbors NOT in compressed set
            let mut rest = Vec::new();
            for &n in &all_nbrs {
                if !compressed_set.contains(&n) {
                    rest.push(n);
                }
            }

            // Set via CompressedGraph's split interface
            self.graph
                .set_neighbors_split(*origin, &compressed, &rest)
                .unwrap();
        }

        self.graph.update_max_compressed_degree();

        // Phase B (sequential): Connectivity repair via BFS
        log::info!("Phase B: Connectivity repair...");
        let connectivity_start = Instant::now();
        self.repair_connectivity();
        log::info!(
            "Connectivity repair: {:.2}s",
            connectivity_start.elapsed().as_secs_f32()
        );

        // Log compression stats
        let mut total_edges = 0usize;
        let mut compressed_edges = 0usize;
        for i in 0..num_nodes as u32 {
            if let Ok(v) = self.graph.read_vertex(i) {
                total_edges += v.degree();
                compressed_edges += v.compressed_degree() as usize;
            }
        }
        if total_edges > 0 {
            log::info!(
                "Merged graph: {} total edges, {} compressed ({:.1}%), avg degree {:.1}, avg compressed degree {:.1}, max compressed degree {}",
                total_edges,
                compressed_edges,
                compressed_edges as f64 / total_edges as f64 * 100.0,
                total_edges as f64 / num_nodes as f64,
                compressed_edges as f64 / num_nodes as f64,
                self.graph.max_compressed_degree(),
            );
        }
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
            max_external_clusters: self.max_external_clusters,
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
        self.max_external_clusters = meta.max_external_clusters;

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
    max_external_clusters: usize,
}
