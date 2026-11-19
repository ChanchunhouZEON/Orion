/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Graph representations and the node-access contract shared by traversal.
//!
//! `phased_graph` owns directly addressable partitioned slabs; `storage_graph`
//! acquires validated node records from immutable object storage. `graph_access`
//! gives traversal a node view whose owner keeps its partition slices alive.
//! This module defines neither search policy nor dataset representations.

pub(crate) mod graph_access;
pub(crate) mod phased_graph;
#[cfg(feature = "experimental-storage")]
pub mod storage_graph;

pub(crate) use phased_graph::PhasedGraph;
