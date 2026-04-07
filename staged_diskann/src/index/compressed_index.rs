/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::algorithm::clustering::{CohesiveClusterManager, INVALID_CLUSTER_AFFILIATION};
use crate::model::FixedChunkPQTable;
use crate::model::scratch::InMemScratchPool;
use crate::utils::DELIMITER_LENGTH;
use diskann::common::{ANNError, ANNResult};
use diskann::model::{CsrGraph, InMemoryGraph, InmemDataset};
#[cfg(feature = "visualization")]
use ndarray::ArcArray2;
use ndarray::{ArcArray1, Array1};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use vector::FullPrecisionDistance;

/// Compressed DiskANN with cluster-aware graph pruning and two-phase search.
/// Const-generic `N` enables SIMD-accelerated distance computation during search reranking.
///
/// Uses a `CsrGraph` with single buffer + bounded write queue. Each node's
/// neighbors are ordered:
/// - First `compressed_degree` neighbors = "compressed" (used in phase 2)
/// - Remaining neighbors = "full" (used only in phase 1)
///
/// Build-only data (`candidate_sets`, `storage_layout`, `cluster_centroids`)
/// is freed immediately after clustering + graph reorder completes.
pub struct StagedDiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Vector data storage taken from the DiskANN `InmemIndex`.
    pub dataset: InmemDataset<f32, N>,
    /// CSR graph with single buffer + write queue for concurrent updates.
    pub graph: CsrGraph,
    pub entry: u32,
    pub num_nodes: usize,
    #[allow(dead_code)]
    pub compressed_graph_save_path: PathBuf,
    pub point_affiliation: ArcArray1<i32>,

    #[cfg(feature = "visualization")]
    pub storage_layout: HashMap<u32, HashSet<u32>>,
    #[cfg(feature = "visualization")]
    cluster_centroids: HashMap<u32, u32>,

    // Pruning parameters
    pub max_connection_clusters: usize,
    pub max_edges_per_cluster: usize,
    pub max_pruned_degree: usize,
    pub max_cluster_point_size: usize,
    pub critical_minimum_rate: f32,

    #[allow(dead_code)]
    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,

    /// Pool of pre-allocated scratch spaces for in-memory search.
    /// Lazily initialized on first call to `search()`.
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,
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
        candidate_sets: Arc<Vec<Vec<u32>>>,
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

        let mut result = StagedDiskANN {
            dataset,
            graph: csr,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            compressed_graph_save_path,
            point_affiliation: Array1::from_elem(num_nodes, -1).to_shared(),
            #[cfg(feature = "visualization")]
            storage_layout: HashMap::new(),
            #[cfg(feature = "visualization")]
            cluster_centroids: HashMap::new(),
            max_connection_clusters,
            max_edges_per_cluster,
            max_pruned_degree,
            max_cluster_point_size,
            critical_minimum_rate,
            is_save,
            inmem_scratch_pool: OnceLock::new(),
        };

        // Run clustering + label+reorder. candidate_sets is consumed (freed
        // when the clustering manager and its ClusterPointManagers are dropped).
        let _ = result.load_or_build_compressed_graph(candidate_sets);

        result
    }

    /// Convenience: build from a `&InMemoryGraph` reference.
    #[allow(clippy::too_many_arguments)]
    pub fn from_graph_ref(
        dataset: InmemDataset<f32, N>,
        inmem_graph: &InMemoryGraph,
        candidate_sets: Arc<Vec<Vec<u32>>>,
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
        let csr = CsrGraph::from_adjacency_list(adj, inmem_graph.max_degree());
        log::info!("  graph_to_csr:   {:.3}s", t_copy.elapsed().as_secs_f32());
        Self::new(
            dataset,
            csr,
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

    /// Convert the dataset to `ArcArray2<f32>` for visualization.
    #[cfg(feature = "visualization")]
    pub fn data_as_arc_array2(&self) -> ArcArray2<f32> {
        let n = self.num_nodes;
        let flat: Vec<f32> = self.dataset.get_data()[..n * N].to_vec();
        ndarray::Array2::from_shape_vec((n, N), flat)
            .expect("shape mismatch")
            .into_shared()
    }

    /// Return per-node (full_degree, compressed_degree) from the CsrGraph.
    /// Used for diagnostics.
    pub fn csr_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.num_nodes;
        (0..n)
            .map(|i| (self.graph.degree(i), self.graph.compressed_degree(i)))
            .collect()
    }

    /// Run clustering to build the compressed graph.
    ///
    /// `candidate_sets` is consumed: freed when the clustering manager and its
    /// `ClusterPointManager`s are dropped at the end of this method.
    fn load_or_build_compressed_graph(
        &mut self,
        candidate_sets: Arc<Vec<Vec<u32>>>,
    ) -> ANNResult<()> {
        log::info!("🏗️: clustering and compressing graph structure  mem={}", diskann::utils::mem_usage());
        let total_start = Instant::now();

        let num_nodes = self.num_nodes;

        let cluster_start = Instant::now();
        let clusters = {
            let mut manager = CohesiveClusterManager::new(
                num_nodes as u32,
                &self.dataset,
                &self.graph,
                &candidate_sets,
                self.max_cluster_point_size,
                self.critical_minimum_rate,
            );

            log::info!("🌎: clustering with CohesiveClustering");
            manager
                .construct_cohesive_clusters()
                .map_err(|_| ANNError::log_cluster_error("Cluster failed!".to_string()))?;
            self.point_affiliation = manager.point_affiliation.clone();
            log::info!("🌎: clustering done, before into_inner  mem={}", diskann::utils::mem_usage());
            manager.cohesive_clusters.into_inner()
        };
        log::info!("🌎: manager dropped  mem={}", diskann::utils::mem_usage());
        let cluster_time = cluster_start.elapsed();
        log::info!(
            "🌎: clustering time (CohesiveClustering): {:.2}s",
            cluster_time.as_secs_f32()
        );

        #[cfg(feature = "visualization")]
        {
            self.cluster_centroids = clusters
                .iter()
                .filter_map(|(&k, v)| v.centroid().map(|c| (k, c)))
                .collect();
            self.storage_layout = clusters
                .iter()
                .map(|(&k, v)| (k, v.cluster_point().keys().copied().collect()))
                .collect();
        }
        drop(clusters);
        log::info!("🌎: clusters dropped  mem={}", diskann::utils::mem_usage());
        drop(candidate_sets);
        log::info!("🌎: candidate_sets dropped  mem={}", diskann::utils::mem_usage());

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

        // 3. Save the compressed graph (visualization builds only)
        #[cfg(feature = "visualization")]
        if self.is_save {
            if let Err(e) = self.save(&self.compressed_graph_save_path.clone()) {
                log::warn!("Failed to save compressed graph: {}", e);
            }
        }

        log::info!("{}", "-".repeat(DELIMITER_LENGTH));
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
    /// `compressed_end`, written directly to the `CsrGraph`.
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
        let graph = &self.graph;
        let max_clusters = self.max_connection_clusters;
        let max_edges = self.max_edges_per_cluster;

        // Parallel: label + reorder per node (lock-free, each thread writes its own node).
        // (E) groups uses stack-allocated fixed array — no heap alloc.
        const MAX_GROUPS: usize = 16; // max_connection_clusters ≤ 16
        (0..num_nodes).into_par_iter().for_each(|origin| {
            let nbrs = graph.neighbors(origin);

            if point_affiliation[origin] == INVALID_CLUSTER_AFFILIATION || nbrs.is_empty() {
                return;
            }

            // (E) Stack-allocated groups: (cluster_aff, first_idx, member_bits, count)
            let mut groups = [(0i32, 0usize, 0u64, 0u32); MAX_GROUPS];
            let mut num_groups: usize = 0;

            for (idx, &nn) in nbrs.iter().enumerate() {
                let aff = point_affiliation[nn as usize];
                if aff == INVALID_CLUSTER_AFFILIATION {
                    continue;
                }
                // Find existing group or add new one.
                let mut found = false;
                for g in &mut groups[..num_groups] {
                    if g.0 == aff {
                        if (g.3 as usize) < max_edges {
                            g.2 |= 1u64 << idx;
                            g.3 += 1;
                        }
                        found = true;
                        break;
                    }
                }
                if !found && num_groups < max_clusters && num_groups < MAX_GROUPS {
                    groups[num_groups] = (aff, idx, 1u64 << idx, 1);
                    num_groups += 1;
                }
            }

            if num_groups == 0 {
                return;
            }

            // Sort groups by first-appearance index.
            groups[..num_groups].sort_unstable_by_key(|g| g.1);

            let mut compressed_bits: u64 = 0;
            for g in &groups[..num_groups] {
                compressed_bits |= g.2;
            }

            // Build reordered neighbor list: compressed first, rest after.
            let mut reordered = Vec::with_capacity(nbrs.len());
            for g in &groups[..num_groups] {
                let mut b = g.2;
                while b != 0 {
                    let idx = b.trailing_zeros() as usize;
                    reordered.push(nbrs[idx]);
                    b &= b - 1;
                }
            }
            let cd = reordered.len();
            for (idx, &n) in nbrs.iter().enumerate() {
                if compressed_bits & (1u64 << idx) == 0 {
                    reordered.push(n);
                }
            }

            // Write reordered neighbors directly (lock-free, each thread owns its node).
            unsafe {
                graph.set_neighbors_reordered_unchecked(origin, &reordered[..cd], &reordered[cd..]);
            }
        });

        log::info!("  label+reorder:  {:.3}s", t_fused.elapsed().as_secs_f32());

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
            for &n in self.graph.neighbors(cur as usize) {
                if reachable.insert(n) {
                    queue.push_back(n);
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
            let neighbors: Vec<u32> = self.graph.neighbors(u as usize).to_vec();
            let mut best_reachable: Option<u32> = None;
            let mut best_dist = f32::MAX;

            for &n in &neighbors {
                if reachable.contains(&n) {
                    let dist = self
                        .dataset
                        .get_distance(u, n, vector::Metric::L2)
                        .unwrap_or(f32::MAX);
                    if dist < best_dist {
                        best_dist = dist;
                        best_reachable = Some(n);
                    }
                }
            }

            if let Some(n) = best_reachable {
                // Add edge u -> n (append to rest portion)
                let u_cd = self.graph.compressed_degree(u as usize);
                if !self.graph.contains_edge(u, n) {
                    let compressed: Vec<u32> = self.graph.compressed_neighbors(u as usize).to_vec();
                    let mut rest: Vec<u32> = self.graph.neighbors(u as usize)[u_cd..].to_vec();
                    rest.push(n);
                    self.graph.update_node(u as usize, &compressed, &rest);
                }
                // Add reverse edge n -> u
                let n_cd = self.graph.compressed_degree(n as usize);
                if !self.graph.contains_edge(n, u) {
                    let compressed: Vec<u32> = self.graph.compressed_neighbors(n as usize).to_vec();
                    let mut rest: Vec<u32> = self.graph.neighbors(n as usize)[n_cd..].to_vec();
                    rest.push(u);
                    self.graph.update_node(n as usize, &compressed, &rest);
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

        let graph_map: HashMap<u32, Vec<u32>> = (0..self.num_nodes)
            .map(|i| (i as u32, self.graph.neighbors(i).to_vec()))
            .collect();
        let compressed_map: HashMap<u32, Vec<u32>> = (0..self.num_nodes)
            .map(|i| (i as u32, self.graph.compressed_neighbors(i).to_vec()))
            .collect();

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
        visualization::draw_compressed_graph(&data, &self.storage_layout, &compressed_map, &path)?;

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
        let dir = path.as_ref().parent().unwrap_or(Path::new("."));
        fs::create_dir_all(dir)?;

        // 1. Save CsrGraph binary
        let graph_path = path.as_ref().with_extension("csrgraph");
        self.graph.save(&graph_path)?;

        // 2. Save metadata via bincode
        let meta = CompressedDiskANNMeta {
            version: 4,
            entry: self.entry,
            point_affiliation: self.point_affiliation.to_vec(),
            max_pruned_degree: self.max_pruned_degree,
            max_connection_clusters: self.max_connection_clusters,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut writer, config)?;
        writer.flush()?;
        Ok(())
    }

    fn load<P: AsRef<Path> + ?Sized>(&mut self, path: &P) -> anyhow::Result<()> {
        let graph_path = path.as_ref().with_extension("csrgraph");
        if !graph_path.exists() {
            anyhow::bail!("CsrGraph file not found: {:?}", graph_path);
        }
        self.graph = CsrGraph::load(&graph_path)?;

        let file = File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: CompressedDiskANNMeta =
            bincode::serde::decode_from_std_read(&mut reader, config)?;
        self.entry = meta.entry;
        self.point_affiliation = ndarray::Array1::from_vec(meta.point_affiliation).into_shared();
        self.max_pruned_degree = meta.max_pruned_degree;
        self.max_connection_clusters = meta.max_connection_clusters;
        self.num_nodes = self.graph.num_nodes();
        Ok(())
    }
}

/// Serializable metadata for CompressedDiskANN (separate from CsrGraph binary).
#[derive(serde::Serialize, serde::Deserialize)]
struct CompressedDiskANNMeta {
    version: u32,
    entry: u32,
    point_affiliation: Vec<i32>,
    max_pruned_degree: usize,
    max_connection_clusters: usize,
}
