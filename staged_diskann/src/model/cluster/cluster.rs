/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::sorted_small::SortedSmallSet;

/// A point within a cohesive cluster, tracking its relationships.
/// All sets are bounded by `MAX_CLUSTER_CAP` (16) and stored inline.
#[derive(Debug, Clone)]
pub struct ClusterPoint {
    /// Points whose candidate sets contain this point.
    pub in_candidate_set: SortedSmallSet,
    /// Cluster points that appear in this point's candidate set.
    pub cluster_point_in_cur_candidates: SortedSmallSet,
    /// Bidirectional graph neighbors within the cluster.
    pub connected_set: SortedSmallSet,
}

impl ClusterPoint {
    pub fn new(
        in_candidate_set: SortedSmallSet,
        cluster_point_in_cur_candidates: SortedSmallSet,
        connected_set: SortedSmallSet,
    ) -> Self {
        Self {
            in_candidate_set,
            cluster_point_in_cur_candidates,
            connected_set,
        }
    }

    /// Create an empty ClusterPoint with only self in `in_candidate_set`.
    pub fn new_with_self(id: u32) -> Self {
        Self {
            in_candidate_set: SortedSmallSet::with_one(id),
            cluster_point_in_cur_candidates: SortedSmallSet::new(),
            connected_set: SortedSmallSet::new(),
        }
    }
}
