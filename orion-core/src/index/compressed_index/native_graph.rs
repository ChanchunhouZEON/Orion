/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Stream codec shared by typed and runtime-dimensional graphs.
use super::{CompressedIndex, IndexData};
use crate::model::PhasedGraph;
use crate::{Error, Result};

use std::io::{BufWriter, Read, Write};

impl<D: IndexData, S: Default> CompressedIndex<D, S> {
    /// Return per-node (degree, local_count) from the PhasedGraph.
    #[cfg(feature = "benchmark")]
    pub fn graph_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.num_nodes;
        (0..n)
            .map(|i| (self.graph.degree(i), self.graph.local_count(i)))
            .collect()
    }

    /// Check shape without scanning a potentially billion-node graph.
    /// Graph edge validity remains the graph producer's responsibility.
    pub(crate) fn validate_dataset_graph(&self) -> Result<()> {
        let rows = self.dataset.point_count();
        if rows == 0
            || rows > u32::MAX as usize
            || self.dataset.dimension() == 0
            || rows != self.graph.num_nodes()
            || rows != self.num_nodes
            || self.entry as usize >= rows
        {
            return Err(Error::cache(format!(
                "incompatible dataset/graph: rows={rows}, dimension={}, graph_nodes={}, entry={}",
                self.dataset.dimension(),
                self.graph.num_nodes(),
                self.entry
            )));
        }
        Ok(())
    }

    /// Encode graph metadata and the resident slab into separate streams.
    /// The caller owns buffering, flush, and publication of both outputs.
    pub fn encode_native_graph(
        &self,
        mut metadata: impl Write,
        graph: &mut BufWriter<impl Write>,
    ) -> Result<()> {
        self.validate_dataset_graph()?;
        let meta = OrionMeta {
            version: 6,
            entry: self.entry,
        };
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut metadata, config)?;
        let base_shape = GraphDatasetShape {
            dimension: self.dataset.dimension() as u64,
            points: self.dataset.point_count() as u64,
            element_type: self.dataset.element_type().to_owned(),
        };
        bincode::serde::encode_into_std_write(&base_shape, &mut metadata, config)?;
        self.graph.encode(graph)
    }

    /// Decode native graph streams directly into their final slab allocation.
    /// Validates shape, byte length and edges. The caller verifies content identity.
    pub fn decode_native_graph(
        dataset: D,
        reader: impl Read,
        graph: impl Read,
        graph_bytes: u64,
    ) -> Result<Self> {
        Self::decode_native_graph_with_storage(
            dataset,
            reader,
            graph,
            graph_bytes,
            &Default::default(),
        )
    }

    pub fn decode_native_graph_with_storage(
        dataset: D,
        mut reader: impl Read,
        graph: impl Read,
        graph_bytes: u64,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: OrionMeta = bincode::serde::decode_from_std_read(&mut reader, config)?;
        // Reject incompatible data before allocating/loading the graph payload.
        match meta.version {
            6 => {
                let base_shape: GraphDatasetShape =
                    bincode::serde::decode_from_std_read(&mut reader, config.with_limit::<4096>())?;
                if base_shape.dimension != dataset.dimension() as u64
                    || base_shape.points != dataset.point_count() as u64
                    || base_shape.element_type != dataset.element_type()
                {
                    return Err(Error::cache(
                        "cached graph dataset dimension, count or element type does not match",
                    )
                    .at("load_cache")
                    .with_detail("expected_dimension", dataset.dimension())
                    .with_detail("actual_dimension", base_shape.dimension)
                    .with_detail("expected_points", dataset.point_count())
                    .with_detail("actual_points", base_shape.points)
                    .with_detail("expected_element_type", dataset.element_type())
                    .with_detail("actual_element_type", base_shape.element_type));
                }
            }
            _ => {
                return Err(Error::cache(format!(
                    "unsupported or unverifiable graph metadata version {}",
                    meta.version
                )));
            }
        }
        let graph = PhasedGraph::decode_with_storage(graph, graph_bytes, storage)?;
        let idx = Self::from_phased_graph(dataset, graph, meta.entry);
        idx.validate_dataset_graph()?;
        Ok(idx)
    }
}

/// Serializable metadata for Orion (separate from PhasedGraph binary).
#[derive(serde::Serialize, serde::Deserialize)]
struct OrionMeta {
    version: u32,
    entry: u32,
}

/// Serialized description of the base associated with a graph cache. This is
/// separate from the public quantized-dataset construction input: graph reuse
/// also requires the coordinate type to match. Fixed-width counts are part of
/// the cache format; SIMD padding is deliberately excluded.
#[derive(serde::Serialize, serde::Deserialize)]
struct GraphDatasetShape {
    dimension: u64,
    points: u64,
    element_type: String,
}
