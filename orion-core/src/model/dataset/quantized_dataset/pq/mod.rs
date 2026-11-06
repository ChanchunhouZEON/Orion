/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Slice-based product quantization, owned by Orion. No ndarray or DiskANN PQ
//! implementation is used. Training is deterministic and bounded by a sample;
//! encoding visits logical rows without making a packed copy of the base.
//!
//! ```
//! use orion_ann_core::model::dataset::quantized_dataset::pq::FixedChunkPQTable;
//! let rows = [0., 1., 2., 3., 4., 5., 6., 7.];
//! let table = FixedChunkPQTable::train(&rows, 2, 4, 2)?;
//! assert_eq!(table.encode(&rows, 2)?.len(), 4);
//! # Ok::<(), orion_ann_core::Error>(())
//! ```
mod codebook;
mod codec;
mod dataset;
mod storage;
pub use codebook::{FixedChunkPQTable, NUM_PQ_CENTROIDS, PqBuildOptions};
pub use dataset::{PqDataset, PqMetric, PqQuery};
pub use storage::PqCodeStorage;
#[cfg(test)]
mod tests;
