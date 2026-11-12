/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Experimental graph import boundary. Suppliers own partitioning and graph
//! quality; Orion validates structure without sorting, pruning or adding edges.
//! No supplier runs inside the search loop. Node IDs refer to base row order.

mod import;
mod source;
mod spec;
mod staged;

use import::validate_row;
pub use source::{NodePartitions, OwnedPartitions, PartitionSource, SlicePartitions};
pub use spec::{PartitionElement, PartitionMetric, PartitionOrigin, PartitionSpec};
pub use staged::{StagedReader, write_staged};
