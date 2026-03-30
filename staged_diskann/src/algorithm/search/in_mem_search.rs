/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::model::{Neighbor, NeighborPriorityQueue};
use crate::utils::l2_distance;
use crate::{DistanceConvergenceChecker, StagedDiskANN};
use diskann::common::ANNResult;
use ndarray::ArrayView1;
use std::collections::HashSet;
use vector::FullPrecisionDistance;

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Search algorithm
    pub fn search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<Vec<u32>> {
        let query = ArrayView1::from(query);
        let mut dcc = DistanceConvergenceChecker::new(window_size, epsilon);
        let entry = self.entry;
        let mut visited = HashSet::<u32>::new();

        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(
            entry,
            l2_distance(query, self.data.row(entry as usize)),
        ));

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            visited.insert(neighbor.id);

            // Read vertex from the CompressedGraph (single RwLock read)
            if let Ok(vertex) = self.graph.read_vertex(neighbor.id) {
                // Switch between full neighbors (Phase 1) and compressed neighbors (Phase 2)
                let neighbors_to_use = if !dcc.update(neighbor.distance) {
                    // Phase 1: use ALL neighbors
                    vertex.get_neighbors()
                } else {
                    // Phase 2: use only compressed neighbors
                    vertex.get_compressed_neighbors()
                };

                for &nn in neighbors_to_use {
                    neighbor_pq.insert(Neighbor::new(
                        nn,
                        l2_distance(query, self.data.row(nn as usize)),
                    ));
                }
            }
        }

        Ok(neighbor_pq
            .neighbors()
            .iter()
            .map(|neighbor| neighbor.id)
            .clone()
            .take(k)
            .collect())
    }
}
