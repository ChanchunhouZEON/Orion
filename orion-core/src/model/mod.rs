/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

pub mod dataset;
pub mod neighbor;
pub(crate) mod phased_graph;
pub mod scratch;
// Shared output models; the crate root exposes the supported public types.
pub(crate) mod search_result;
pub mod visited_set;

pub use dataset::quantized_dataset::pq::{FixedChunkPQTable, NUM_PQ_CENTROIDS};
pub use dataset::{
    JLSparseDataset, JLSparseDatasetMips, L2KTDataset, L2U8, L2U16, MipsI8, MipsI16, PreparedQuery,
    QuantParamsL2, QuantParamsMips, QuantSpec, QuantizedDataset, QuantizedDatasetCompute,
    QuantizedDatasetShape, ResidentMemory, ScalarQuantizedDataset,
};
pub use neighbor::{Neighbor, NeighborPriorityQueue};
pub(crate) use phased_graph::PhasedGraph;
pub use scratch::{InMemScratchGuard, InMemScratchPool, InMemSearchScratch};
