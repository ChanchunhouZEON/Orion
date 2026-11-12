/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validated partition and STAG construction into the final graph slab.
use crate::index::{CompressedIndex, IndexData};
use crate::model::PhasedGraph;
use crate::{Error, Result};
use std::io::Read;
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
        Self::from_staged_graph_with_storage(
            dataset,
            reader,
            byte_len,
            max_degree,
            entry,
            &Default::default(),
        )
    }

    pub fn from_staged_graph_with_storage(
        dataset: D,
        reader: impl Read,
        byte_len: u64,
        max_degree: u32,
        entry: u32,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let nodes = dataset.point_count();
        Self::validate_graph_shape(&dataset, nodes, entry)?;
        let graph =
            PhasedGraph::decode_staged_with_storage(reader, byte_len, nodes, max_degree, storage)?;
        Ok(Self::from_phased_graph(dataset, graph, entry))
    }

    pub(crate) fn validate_graph_shape(dataset: &D, nodes: usize, entry: u32) -> Result<()> {
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

impl<const N: usize, T> crate::StaticOrion<N, T>
where
    [T; N]: orion_ann_vector::FullPrecisionDistance<T, N>,
{
    /// Build a static index from pre-computed partitions.
    ///
    /// # Panics
    /// Panics for invalid graph shape, neighbor IDs, or allocation failure. Prefer
    /// [`Self::from_graph_partitions`] for fallible construction.
    pub fn new(
        dataset: orion_ann_diskann::model::InmemDataset<T, N>,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
    ) -> Self {
        let num_nodes = partitions.len();
        Self::validate_graph_shape(&dataset, num_nodes, entry)
            .expect("invalid graph shape or entry");

        let t_graph = std::time::Instant::now();
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
