/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Audited graph boundary. Storage and mutable graph access remain internal.
use super::{CompressedIndex, IndexData};
use crate::{Error, PhasedGraph, Result};
use std::io::Read;

/// Borrowed, read-only graph inspection through an [`Orion`](crate::Orion) index.
///
/// This view neither locks nor copies the graph. Its lifetime prevents exclusive
/// mutation of the owning index while any returned adjacency slice is alive.
/// The graph owner is deliberately not part of the public API:
/// ```compile_fail
/// use orion_ann_core::PhasedGraph;
/// ```
/// ```compile_fail
/// use orion_ann_core::model::phased_graph::PhasedGraph;
/// ```
/// Nor can callers obtain mutable storage through the owning index:
/// ```compile_fail
/// fn replace_graph(index: &mut orion_ann_core::Orion) {
///     let graph = &mut index.graph;
/// }
/// ```
///
/// Node accessors panic for out-of-range node IDs; use `num_nodes()` to validate
/// external IDs. No method exposes mutable storage or an update queue.
#[derive(Clone, Copy, Debug)]
pub struct GraphView<'a> {
    graph: &'a PhasedGraph,
}

impl<'a> GraphView<'a> {
    fn check_node(&self, node: usize) {
        assert!(node < self.graph.num_nodes(), "graph node ID out of range");
    }

    /// Whether a navigation edge exists; reranking-only extras are excluded.
    pub fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.check_node(from as usize);
        self.check_node(to as usize);
        self.graph.contains_edge(from, to)
    }

    /// Number of graph nodes, matching the index's vector count at construction.
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }
    /// Fixed node stride, in `u32` words, including header and padding.
    pub fn stride(&self) -> usize {
        self.graph.stride()
    }
    /// Maximum navigation degree configured for this graph.
    pub fn max_degree(&self) -> u32 {
        self.graph.max_degree
    }
    /// Recorded global local-neighbor threshold.
    pub fn base_local_count(&self) -> u32 {
        self.graph.base_local_count
    }
    /// Immutable native slab bytes for inspection or memory pinning.
    /// This is an implementation-specific cache payload, not a portable format.
    pub fn buffer_bytes(&self) -> &'a [u8] {
        self.graph.buffer_bytes()
    }
    /// Navigation neighbors in local-then-remote order.
    #[inline]
    pub fn neighbors(&self, node: usize) -> &'a [u32] {
        self.check_node(node);
        self.graph.neighbors(node)
    }
    /// Local navigation partition.
    #[inline]
    pub fn local_neighbors(&self, node: usize) -> &'a [u32] {
        self.check_node(node);
        self.graph.local_neighbors(node)
    }
    /// Remote navigation partition.
    #[inline]
    pub fn remote_neighbors(&self, node: usize) -> &'a [u32] {
        self.check_node(node);
        self.graph.remote_neighbors(node)
    }
    /// Reranking-only candidates.
    #[inline]
    pub fn extra_candidates(&self, node: usize) -> &'a [u32] {
        self.check_node(node);
        self.graph.extra_candidates(node)
    }
    /// Number of navigation neighbors.
    pub fn degree(&self, node: usize) -> usize {
        self.check_node(node);
        self.graph.degree(node)
    }
    /// Number of local neighbors.
    pub fn local_count(&self, node: usize) -> usize {
        self.check_node(node);
        self.graph.local_count(node)
    }
    /// Number of reranking-only candidates.
    pub fn extra_count(&self, node: usize) -> usize {
        self.check_node(node);
        self.graph.extra_count(node)
    }
}

impl<D, S> CompressedIndex<D, S> {
    /// Inspect graph topology without exposing the internal graph owner.
    #[inline]
    pub fn graph(&self) -> GraphView<'_> {
        GraphView { graph: &self.graph }
    }
}

impl<D: IndexData, S: Default> CompressedIndex<D, S> {
    /// Import validated `(local, remote, extra)` partitions into an index.
    ///
    /// The dataset is moved without copying. Graph data is copied once into the
    /// final slab. Calibration and persistence are explicit separate operations.
    ///
    /// # Errors
    /// Rejects mismatched dataset/graph counts, invalid entry or neighbor IDs,
    /// oversized partitions and allocation failures before publishing an index.
    ///
    /// # Example
    /// ```
    /// use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};
    /// use orion_ann_vector::L2F32Distance;
    /// let data = PaddedDataset::from_packed(
    ///     &[0.0f32, 1.0, 2.0, 3.0], PaddedLayout::for_kernel::<L2F32Distance>(2)?,
    /// )?;
    /// let parts = [(vec![1], vec![], vec![]), (vec![0], vec![], vec![])];
    /// let index = Orion::from_graph_partitions(data, &parts, 0, 1, 0)?;
    /// assert_eq!(index.graph().neighbors(0), &[1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_graph_partitions(
        dataset: D,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
    ) -> Result<Self> {
        Self::validate_graph_shape(&dataset, partitions.len(), entry)?;
        let graph = PhasedGraph::try_from_partitions(partitions, max_degree, max_extra)?;
        Ok(Self::from_phased_graph(dataset, graph, entry))
    }

    /// Stream a ParlayANN STAG v3 graph into an index without an intermediate graph.
    ///
    /// `entry` explicitly selects the search entry point; the file's hint is not
    /// substituted. Dataset order must match the exported graph's node IDs.
    ///
    /// # Errors
    /// Rejects invalid entry/counts/IDs, mismatched graph degree, malformed or
    /// truncated files, allocation failures and filesystem errors. Data identity
    /// is the caller's responsibility; this does not authenticate the artifact.
    pub fn from_staged_graph(
        dataset: D,
        reader: impl Read,
        byte_len: u64,
        max_degree: u32,
        entry: u32,
    ) -> Result<Self> {
        let nodes = dataset.point_count();
        Self::validate_graph_shape(&dataset, nodes, entry)?;
        let graph = PhasedGraph::decode_staged(reader, byte_len, nodes, max_degree)?;
        Ok(Self::from_phased_graph(dataset, graph, entry))
    }

    pub(super) fn validate_graph_shape(dataset: &D, nodes: usize, entry: u32) -> Result<()> {
        if nodes == 0 || nodes != dataset.point_count() || entry as usize >= nodes {
            return Err(
                Error::invalid_argument("graph shape or entry does not match dataset")
                    .at("import_graph")
                    .with_detail("dataset_points", dataset.point_count())
                    .with_detail("graph_nodes", nodes)
                    .with_detail("entry", entry),
            );
        }
        Ok(())
    }
}
