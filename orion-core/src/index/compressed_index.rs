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
//! use orion_ann_diskann::model::PaddedDataset;
//! use orion_ann_core::{Orion, DynamicCascade, DynamicSearchConfig, CalibrationConfig};
//! use orion_ann_core::algorithm::search::stage::{admission::L2U8Admission, rerank::DynamicF32Rerank};
//! use orion_ann_vector::{L2F32Distance, PaddedLayout};
//!
//! let base = PaddedDataset::from_packed(
//!     &[1.0f32,2.,3., 4.,5.,6., 7.,8.,9.],
//!     PaddedLayout::for_kernel::<L2F32Distance>(3)?,
//! )?;
//! let parts = [
//!     (vec![1,2],vec![],vec![]), (vec![0,2],vec![],vec![]), (vec![0,1],vec![],vec![]),
//! ];
//! let index = Orion::from_graph_partitions(base, &parts, 0, 2, 0)?;
//! let query = [1.0,2.,3.];
//! let params = index.calibrate(&query,3,3,CalibrationConfig{k:1,..Default::default()})?;
//! let admission = L2U8Admission::new(index.ensure_quantized_dataset());
//! let rerank = DynamicF32Rerank::<L2F32Distance>::new(index.dataset());
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
use orion_ann_diskann::model::InmemDataset;
use orion_ann_vector::FullPrecisionDistance;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

mod access;
mod calibration;
mod cascade;
mod data;
mod dynamic;
pub mod graph;
mod native_graph;
mod sidecars;
mod subset;
pub use cascade::DynamicCascade;
pub use data::IndexData;
pub use dynamic::{DynamicElement, DynamicSearchConfig};
pub use sidecars::DynamicSidecars;
#[doc(hidden)]
pub use sidecars::{DecodeSidecar, EncodeSidecar, SidecarCache};

/// Const-generic compatibility index for legacy benchmarks and core tests.
/// All application frontends use the runtime-dimensional [`Orion`].
pub type StaticOrion<const N: usize, T = f32> =
    CompressedIndex<InmemDataset<T, N>, DynamicSidecars>;
/// Primary runtime-dimensional index: padded storage, quantized cascades and
/// automatic calibration. `StaticOrion<N,T>` remains a compatibility adapter.
pub type Orion<T = f32> = CompressedIndex<orion_ann_diskann::model::PaddedDataset<T>>;

/// Orion index with PhasedGraph-based two-phase search.
///
/// Uses a `PhasedGraph` where each node's neighbors are partitioned:
/// - `local_neighbors`: candidate-set-confirmed (navigation + reranking)
/// - `remote_neighbors`: long-range shortcuts (navigation only)
/// - `extra_candidates`: non-graph candidate-set points (reranking only)
pub struct CompressedIndex<D, S = DynamicSidecars> {
    /// Owned vector storage; moving it into the index does not copy the payload.
    pub(crate) dataset: D,
    /// Validated slab with local/remote/extra partitions.
    pub(crate) graph: PhasedGraph,
    pub(crate) entry: u32,
    pub(crate) num_nodes: usize,
    // PQ relevant
    pub(crate) pq: Option<Arc<FixedChunkPQTable>>,
    pub(crate) pq_codes: Option<Vec<u8>>,
    pub(crate) num_pq_chunks: Option<usize>,

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
    ///
    /// # Panics
    /// Panics for invalid graph shape, neighbor IDs, or allocation failure. Prefer
    /// [`Self::from_graph_partitions`] for fallible construction.
    pub fn new(
        dataset: InmemDataset<T, N>,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
    ) -> Self {
        let num_nodes = partitions.len();
        Self::validate_graph_shape(&dataset, num_nodes, entry)
            .expect("invalid graph shape or entry");

        let t_graph = Instant::now();
        let graph = PhasedGraph::build_from_partitions(partitions, max_degree, max_extra);
        log::info!(
            "🏗️: PhasedGraph built in {:.3}s  mem={}",
            t_graph.elapsed().as_secs_f32(),
            orion_ann_diskann::utils::mem_usage(),
        );
        graph.print_stats();

        Self::from_phased_graph(dataset, graph, entry)
    }
}

impl<D, S: Default> CompressedIndex<D, S> {
    /// Build from a pre-built PhasedGraph (e.g. cloned from another instance).
    pub(crate) fn from_phased_graph(dataset: D, graph: PhasedGraph, entry: u32) -> Self {
        let num_nodes = graph.num_nodes();
        Self {
            dataset,
            graph,
            entry,
            num_nodes,
            pq: None,
            pq_codes: None,
            num_pq_chunks: None,
            inmem_scratch_pool: OnceLock::new(),
            sidecars: S::default(),
        }
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<D, S> std::fmt::Debug for CompressedIndex<D, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressedIndex")
            .field("storage_type", &(std::any::type_name::<D>()))
            .field("num_nodes", &self.num_nodes)
            .field("entry", &self.entry)
            .field("graph", &self.graph)
            .field(
                "scratch_initialized",
                &self.inmem_scratch_pool.get().is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl<D: IndexData, S> CompressedIndex<D, S> {
    /// Attach fitted product-quantization data to an owned index.
    /// # Errors
    /// Rejects incompatible dimensions or packed code lengths. This is an
    /// experimental PQ search facility, separate from scalar-quantized cascades.
    pub fn with_product_quantization(
        mut self,
        table: Arc<FixedChunkPQTable>,
        codes: Vec<u8>,
    ) -> crate::Result<Self> {
        let chunks = table.get_num_chunks();
        if table.dim() != self.dataset.dimension()
            || chunks == 0
            || self.num_nodes.checked_mul(chunks) != Some(codes.len())
        {
            return Err(crate::Error::invalid_argument(
                "PQ table/codes do not match the base shape",
            ));
        }
        self.num_pq_chunks = Some(chunks);
        self.pq = Some(table);
        self.pq_codes = Some(codes);
        Ok(self)
    }
}
