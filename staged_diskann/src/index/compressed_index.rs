/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::FixedChunkPQTable;
use crate::model::PhasedGraph;
use crate::model::QuantizedDataset;
use crate::model::scratch::InMemScratchPool;
use diskann::model::InmemDataset;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock}; // Arc still needed for pq field
use std::time::Instant;
use vector::FullPrecisionDistance;

/// Staged DiskANN with PhasedGraph-based two-phase search.
///
/// Uses a `PhasedGraph` where each node's neighbors are partitioned:
/// - `local_neighbors`: candidate-set-confirmed (navigation + reranking)
/// - `remote_neighbors`: long-range shortcuts (navigation only)
/// - `extra_candidates`: non-graph candidate-set points (reranking only)
pub struct StagedDiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Vector data storage taken from the DiskANN `InmemIndex`.
    pub dataset: InmemDataset<f32, N>,
    /// PhasedGraph with local/remote/extra layout + bidir bits.
    pub graph: PhasedGraph,
    pub entry: u32,
    pub num_nodes: usize,
    #[allow(dead_code)]
    pub compressed_graph_save_path: PathBuf,

    #[allow(dead_code)]
    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,

    /// Pool of pre-allocated scratch spaces for in-memory search.
    /// Lazily initialized on first call to `search()`.
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,

    /// U8-quantized copy of `dataset` for low-precision pre-filter during
    /// `search` (ParlayANN-style). Built lazily on first access
    /// via `ensure_quantized_dataset`.
    pub(crate) q_dataset: OnceLock<QuantizedDataset<N>>,
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build a StagedDiskANN from pre-computed partitions.
    pub fn new(
        dataset: InmemDataset<f32, N>,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
        compressed_graph_save_path: Option<PathBuf>,
        is_save: bool,
    ) -> Self {
        let num_nodes = partitions.len();

        let num_pq_chunks = pq.as_ref().map(|p| p.get_num_chunks());

        let compressed_graph_save_path = compressed_graph_save_path.unwrap_or_else(|| {
            let dir = PathBuf::from("staged_diskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!("staged_diskann_n{}.bin", num_nodes))
        });

        let t_graph = Instant::now();
        let graph = PhasedGraph::build_from_partitions(partitions, max_degree, max_extra);
        log::info!(
            "🏗️: PhasedGraph built in {:.3}s  mem={}",
            t_graph.elapsed().as_secs_f32(),
            diskann::utils::mem_usage(),
        );
        graph.print_stats();

        let result = StagedDiskANN {
            dataset,
            graph,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            compressed_graph_save_path,
            is_save,
            inmem_scratch_pool: OnceLock::new(),
            q_dataset: OnceLock::new(),
        };

        if is_save {
            if let Err(e) = result.save(&result.compressed_graph_save_path.clone()) {
                log::warn!("Failed to save graph: {}", e);
            }
        }

        result
    }

    /// Build from a pre-built PhasedGraph (e.g. cloned from another instance).
    pub fn from_phased_graph(
        dataset: InmemDataset<f32, N>,
        graph: PhasedGraph,
        entry: u32,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
    ) -> Self {
        let num_nodes = graph.num_nodes();
        let num_pq_chunks = pq.as_ref().map(|p| p.get_num_chunks());
        StagedDiskANN {
            dataset,
            graph,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            compressed_graph_save_path: PathBuf::from(""),
            is_save: false,
            inmem_scratch_pool: OnceLock::new(),
            q_dataset: OnceLock::new(),
        }
    }

    /// Lazily obtain the u8 quantized dataset. Tries the sidecar `.qds`
    /// file next to the cache path first (O(ms) memcpy); on miss, builds
    /// from the f32 dataset (O(N × dim) scan + scale) and writes it back
    /// so subsequent runs hit the fast path.
    pub fn ensure_quantized_dataset(&self) -> &QuantizedDataset<N> {
        self.q_dataset.get_or_init(|| {
            let qds_path = self.compressed_graph_save_path.with_extension("qds");
            if qds_path.exists() {
                match QuantizedDataset::<N>::load(&qds_path) {
                    Ok(qds) => {
                        log::info!("QuantizedDataset loaded from {:?}", qds_path);
                        return qds;
                    }
                    Err(e) => log::warn!("QuantizedDataset load failed ({e}), rebuilding"),
                }
            }
            let t = Instant::now();
            let qds = QuantizedDataset::from_f32_dataset(&self.dataset);
            log::info!(
                "QuantizedDataset built in {:.2}s",
                t.elapsed().as_secs_f32()
            );
            if !self.compressed_graph_save_path.as_os_str().is_empty() {
                if let Err(e) = qds.save(&qds_path) {
                    log::warn!("QuantizedDataset save failed: {e}");
                } else {
                    log::info!("QuantizedDataset saved to {:?}", qds_path);
                }
            }
            qds
        })
    }

    /// Return per-node (degree, local_count) from the PhasedGraph.
    pub fn graph_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.num_nodes;
        (0..n)
            .map(|i| (self.graph.degree(i), self.graph.local_count(i)))
            .collect()
    }

    // --- IO ---

    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let dir = path.as_ref().parent().unwrap_or(Path::new("."));
        fs::create_dir_all(dir)?;

        let graph_path = path.as_ref().with_extension("pgraph");
        self.graph.save(&graph_path)?;

        let meta = StagedDiskANNMeta {
            version: 5,
            entry: self.entry,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut writer, config)?;
        writer.flush()?;
        Ok(())
    }

    /// Construct from cached PhasedGraph + metadata on disk.
    /// `path` is the metadata `.bin` file; PhasedGraph is read from `path.with_extension("pgraph")`.
    /// The returned instance has an empty `dataset` — caller must install it afterwards.
    pub fn load_from_cache<P: AsRef<Path>>(
        path: P,
        dataset: InmemDataset<f32, N>,
    ) -> anyhow::Result<Self> {
        let graph_path = path.as_ref().with_extension("pgraph");
        if !graph_path.exists() {
            anyhow::bail!("PhasedGraph file not found: {:?}", graph_path);
        }
        let graph = PhasedGraph::load(&graph_path)?;

        let file = File::open(&path)?;
        let mut reader = std::io::BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: StagedDiskANNMeta = bincode::serde::decode_from_std_read(&mut reader, config)?;
        let mut idx = Self::from_phased_graph(dataset, graph, meta.entry, None, None);
        // Preserve the cache path so `ensure_quantized_dataset` can locate
        // the `.qds` sidecar next to it (load fast path).
        idx.compressed_graph_save_path = path.as_ref().to_path_buf();
        Ok(idx)
    }
}

/// Serializable metadata for StagedDiskANN (separate from PhasedGraph binary).
#[derive(serde::Serialize, serde::Deserialize)]
struct StagedDiskANNMeta {
    version: u32,
    entry: u32,
}
