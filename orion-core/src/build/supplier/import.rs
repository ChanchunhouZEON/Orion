/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Audited structural validation and construction of the final graph slab.
use super::{NodePartitions, PartitionElement, PartitionSource, PartitionSpec};
use crate::index::IndexData;
use crate::{CompressedIndex, Error, PhasedGraph, Result};
use orion_ann_platform::region::RegionOptions;

// Reuse O(degree + extras) validation scratch across the whole import. Sorting
// this scratch does not alter supplier order in the final slab.
pub(super) fn validate_row(
    spec: &PartitionSpec,
    node: u32,
    row: NodePartitions<'_>,
    scratch: &mut Vec<u32>,
) -> Result<()> {
    let navigation = row.local.len().checked_add(row.remote.len());
    if navigation.is_none_or(|n| n > spec.max_degree as usize)
        || row.extra.len() > spec.max_extra as usize
    {
        return Err(
            Error::invalid_argument("partition exceeds declared capacity")
                .with_detail("node", node),
        );
    }
    scratch.clear();
    for &id in row.local.iter().chain(row.remote).chain(row.extra) {
        if id >= spec.points || id == node {
            return Err(
                Error::invalid_argument("self edge or out-of-range neighbor")
                    .with_detail("node", node)
                    .with_detail("neighbor", id),
            );
        }
        scratch.push(id);
    }
    scratch.sort_unstable();
    if scratch.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(
            Error::invalid_argument("duplicate neighbor across partitions")
                .with_detail("node", node),
        );
    }
    Ok(())
}

impl<D: IndexData, S: Default> CompressedIndex<D, S> {
    /// Import ordered partitions directly into the final slab, optionally mmap.
    ///
    /// # Errors
    /// Rejects incompatible shape/type, invalid metadata/edges, missing or extra
    /// rows, supplier I/O errors and allocation failures. Metric, preprocessing
    /// and optional base digest are checked by the runtime, which owns base I/O.
    /// Structural validity does not certify connectivity or ANN recall.
    pub fn from_partition_source(
        dataset: D,
        spec: &PartitionSpec,
        source: &mut (impl PartitionSource + ?Sized),
        storage: &RegionOptions,
    ) -> Result<Self> {
        spec.validate()?;
        Self::validate_graph_shape(&dataset, spec.points as usize, spec.entry)?;
        let element = match spec.element {
            PartitionElement::F32 => "f32",
            PartitionElement::U8 => "u8",
        };
        if dataset.dimension() != spec.dimension as usize || dataset.element_type() != element {
            return Err(Error::invalid_argument(
                "partition dimension or element type differs from base",
            ));
        }
        let mut graph = PhasedGraph::try_empty_with_storage(
            spec.points as usize,
            spec.max_degree,
            spec.max_extra as usize,
            storage,
        )?;
        let mut scratch = Vec::new();
        for node in 0..spec.points {
            let row = source.next_partition()?.ok_or_else(|| {
                Error::invalid_argument("missing partition row").with_detail("node", node)
            })?;
            validate_row(spec, node, row, &mut scratch)?;
            graph.set_partition(node as usize, row.local, row.remote, row.extra)?;
        }
        if source.next_partition()?.is_some() {
            return Err(Error::invalid_argument("extra partition rows"));
        }
        Ok(Self::from_phased_graph(dataset, graph, spec.entry))
    }
}
