/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! `Orion` owns a runtime-dimensional padded base and shares graph search,
//! sidecars and calibration with the temporary `StaticOrion<N,T>` adapter.
//!
//! Logical rows are used for fitting quantizers, projection matrices and norms.
//! Each encoded row is then padded for its SIMD kernel. The stream retains the
//! existing near-prefetch and burst schedule; file formats store canonical rows
//! so caches can be repadded on a different architecture.
//!
//! ```
//! use diskann::model::PaddedDataset;
//! use orion::{Orion, DynamicCascade, DynamicSearchConfig, CalibrationConfig, PhasedGraph};
//! use orion::algorithm::search::stage::{admission::L2U8Admission, rerank::DynamicF32Rerank};
//! use vector::{L2F32Distance, PaddedLayout};
//!
//! let base = PaddedDataset::from_packed(
//!     &[1.0f32,2.,3., 4.,5.,6., 7.,8.,9.],
//!     PaddedLayout::for_kernel::<L2F32Distance>(3)?,
//! )?;
//! let graph = PhasedGraph::build_from_partitions(&[
//!     (vec![1,2],vec![],vec![]), (vec![0,2],vec![],vec![]), (vec![0,1],vec![],vec![]),
//! ],2,0);
//! let index = Orion::from_phased_graph(base,graph,0,None,None);
//! let query = [1.0,2.,3.];
//! let params = index.calibrate(&query,3,3,CalibrationConfig{k:1,..Default::default()})?;
//! let admission = L2U8Admission::new(index.ensure_quantized_dataset());
//! let rerank = DynamicF32Rerank::<L2F32Distance>::new(&index.dataset);
//! let ids = index.search_cascade(&query, DynamicSearchConfig {
//!     k:1, search_list_size:3, epsilon:params.threshold,
//!     early_exit_limit:params.early_exit_limit,..Default::default()
//! }, DynamicCascade{prefilter:None,admission:&admission,rerank:&rerank})?;
//! assert_eq!(ids, vec![0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::model::FixedChunkPQTable;
use crate::model::PhasedGraph;
use crate::model::scratch::InMemScratchPool;
use diskann::model::InmemDataset;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use vector::FullPrecisionDistance;

mod cache;
mod calibration;
mod cascade;
mod data;
mod dynamic;
mod sidecars;
pub use cascade::DynamicCascade;
pub use data::IndexData;
pub use dynamic::{DynamicElement, DynamicSearchConfig};
pub use sidecars::DynamicSidecars;

/// Const-generic compatibility index for legacy benchmarks and core tests.
/// All application frontends use the runtime-dimensional [`Orion`].
pub type StaticOrion<const N: usize, T = f32> =
    CompressedIndex<InmemDataset<T, N>, DynamicSidecars>;
/// Primary runtime-dimensional index: padded storage, quantized cascades and
/// automatic calibration. `StaticOrion<N,T>` remains a compatibility adapter.
pub type Orion<T = f32> = CompressedIndex<diskann::model::PaddedDataset<T>>;

/// Orion index with PhasedGraph-based two-phase search.
///
/// Uses a `PhasedGraph` where each node's neighbors are partitioned:
/// - `local_neighbors`: candidate-set-confirmed (navigation + reranking)
/// - `remote_neighbors`: long-range shortcuts (navigation only)
/// - `extra_candidates`: non-graph candidate-set points (reranking only)
pub struct CompressedIndex<D, S = DynamicSidecars> {
    /// Owned vector storage; moving it into the index does not copy the payload.
    pub dataset: D,
    /// PhasedGraph with local/remote/extra layout + bidir bits.
    pub graph: PhasedGraph,
    pub entry: u32,
    pub num_nodes: usize,
    /// Base path for on-disk caches of this index. The metadata blob
    /// is written here directly; every sidecar (PhasedGraph `.pgraph`,
    /// quantized datasets `.qds` / `.qdm8` / `.qdm16` / `.qrbq`) is
    /// derived by `path.with_extension(...)`. Empty `PathBuf` means
    /// "in-memory only, do not persist."
    pub cache_base_path: PathBuf,

    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,

    /// Pool of pre-allocated scratch spaces for in-memory search.
    /// Lazily initialized on first call to `search()`.
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,

    pub(crate) sidecars: S,
}

impl<const N: usize, T> StaticOrion<N, T>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Build a static index from pre-computed partitions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: InmemDataset<T, N>,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
        cache_base_path: Option<PathBuf>,
        is_save: bool,
    ) -> Self {
        let num_nodes = partitions.len();

        let cache_base_path = cache_base_path.unwrap_or_else(|| {
            let dir = PathBuf::from("orion_graphs");
            dir.join(format!("orion_n{}.bin", num_nodes))
        });

        let t_graph = Instant::now();
        let graph = PhasedGraph::build_from_partitions(partitions, max_degree, max_extra);
        log::info!(
            "🏗️: PhasedGraph built in {:.3}s  mem={}",
            t_graph.elapsed().as_secs_f32(),
            diskann::utils::mem_usage(),
        );
        graph.print_stats();

        let mut result = Self::from_phased_graph(dataset, graph, entry, pq, pq_codes);
        result.cache_base_path = cache_base_path;
        result.is_save = is_save;

        if is_save && let Err(e) = result.save(&result.cache_base_path) {
            log::error!("Failed to save graph: {}", e);
        }

        result
    }
}

impl<D, S: Default> CompressedIndex<D, S> {
    /// Build from a pre-built PhasedGraph (e.g. cloned from another instance).
    pub fn from_phased_graph(
        dataset: D,
        graph: PhasedGraph,
        entry: u32,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
    ) -> Self {
        let num_nodes = graph.num_nodes();
        let num_pq_chunks = pq.as_ref().map(|p| p.get_num_chunks());
        Self {
            dataset,
            graph,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            cache_base_path: PathBuf::from(""),
            is_save: false,
            inmem_scratch_pool: OnceLock::new(),
            sidecars: S::default(),
        }
    }
}
