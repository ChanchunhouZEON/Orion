/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use diskann::common::ANNResult;
use ndarray::ArcArray1;
use std::collections::{HashMap, HashSet};

/// Result of a clustering algorithm.
pub struct ClusteringResult {
    pub point_affiliation: ArcArray1<i32>,
    pub storage_layout: HashMap<u32, HashSet<u32>>,
    /// Per-cluster centroid point ID (if available from the clustering algorithm).
    /// Used in Algorithm 7 to compute distances to cluster centroids without recomputation.
    pub centroids: HashMap<u32, u32>,
}

/// Abstract clustering strategy.
pub trait ClusteringStrategy {
    fn cluster(&mut self) -> ANNResult<ClusteringResult>;
    fn name(&self) -> &str;
}

/// Which clustering algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusteringMethod {
    Cohesive,
    LabelPropagation,
}
