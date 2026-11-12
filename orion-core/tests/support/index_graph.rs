/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Tiny graph fixtures use only Orion's public, validated import boundary.
use orion_ann_core::CompressedIndex;
use orion_ann_core::index::IndexData;

pub struct GraphFixture {
    partitions: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    degree: u32,
    extras: usize,
}

impl GraphFixture {
    pub fn build_from_partitions(
        parts: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        degree: u32,
        extras: usize,
    ) -> Self {
        Self {
            partitions: parts.to_vec(),
            degree,
            extras,
        }
    }
}

pub fn from_fixture_graph<D: IndexData>(
    dataset: D,
    graph: GraphFixture,
    entry: u32,
) -> CompressedIndex<D> {
    CompressedIndex::from_graph_partitions(
        dataset,
        &graph.partitions,
        entry,
        graph.degree,
        graph.extras,
    )
    .unwrap()
}
