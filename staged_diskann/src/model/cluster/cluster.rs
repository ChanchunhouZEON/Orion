/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::HashSet;

/// A point within a cohesive cluster, tracking its relationships.
#[derive(Debug, Clone)]
pub struct ClusterPoint {
    /// Points whose candidate sets contain this point.
    pub in_candidate_set: HashSet<u32>,
    /// Cluster points that appear in this point's candidate set.
    pub cluster_point_in_cur_candidates: HashSet<u32>,
    /// Bidirectional graph neighbors within the cluster.
    pub connected_set: HashSet<u32>,
}

impl ClusterPoint {
    pub fn new(
        in_candidate_set: HashSet<u32>,
        cluster_point_in_cur_candidates: HashSet<u32>,
        connected_set: HashSet<u32>,
    ) -> Self {
        Self {
            in_candidate_set,
            cluster_point_in_cur_candidates,
            connected_set,
        }
    }
}
