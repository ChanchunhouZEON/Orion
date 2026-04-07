/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::algorithm::SearchProfile;
use crate::model::{Neighbor, NeighborPriorityQueue};
use crate::{DistanceConvergenceChecker, StagedDiskANN};
use diskann::common::ANNResult;
use diskann::model::Vertex;
use std::collections::HashSet;
use std::time::Instant;
use vector::{FullPrecisionDistance, Metric};

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// SSD simulation search algorithm
    pub fn simulate_ssd_search(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<Vec<u32>> {
        // Preprocess query for PQ distance computation
        let mut query_vec = query.to_vec();
        let (pq, _, _) = self.get_unwrapped_pq_component()?;
        pq.preprocess_query(&mut query_vec);
        let pq_dists = pq.populate_chunk_distances(&query_vec);

        let mut dcc = DistanceConvergenceChecker::new(window_size, epsilon);
        let entry = self.entry;
        let mut visited = HashSet::<u32>::new();

        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(entry, self.pq_distance(entry, &pq_dists)?));

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            visited.insert(neighbor.id);

            // Switch between full neighbors (Phase 1) and compressed neighbors (Phase 2)
            let neighbors_to_use = if !dcc.update(neighbor.distance) {
                // Phase 1: use ALL neighbors
                self.graph.neighbors(neighbor.id as usize)
            } else {
                // Phase 2: use only compressed neighbors
                self.graph.compressed_neighbors(neighbor.id as usize)
            };

            for &nn in neighbors_to_use {
                let dist = self.pq_distance(nn, &pq_dists)?;
                neighbor_pq.insert(Neighbor::new(nn, dist));
            }
        }

        // Rerank all visited nodes with exact distance
        let mut result: Vec<(u32, f32)> = visited
            .iter()
            .map(|&id| {
                let dist = {
                    let qv = Vertex::new(query, 0);
                    let v = self.dataset.get_vertex(id).unwrap();
                    qv.compare(&v, Metric::L2)
                };
                (id, dist)
            })
            .collect();

        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(result.iter().map(|(id, _)| *id).take(k).collect())
    }

    /// Search returning metrics: (results, full_ndc, compressed_ndc, compressed_neighbors_in_mem_cnts)
    pub fn simulate_ssd_search_with_metric(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<(Vec<u32>, u32, u32, u32)> {
        let mut query_vec = query.to_vec();
        let (pq, _, _) = self.get_unwrapped_pq_component()?;
        pq.preprocess_query(&mut query_vec);
        let pq_dists = pq.populate_chunk_distances(&query_vec);

        let mut dcc = DistanceConvergenceChecker::new(window_size, epsilon);
        let entry = self.entry;
        let mut visited = HashSet::<u32>::new();

        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(entry, self.pq_distance(entry, &pq_dists)?));

        let mut compressed_ndc: u32 = 0;
        let mut compressed_neighbors_in_mem_cnts: u32 = 0;
        let mut compressed_neighbors_in_mem = HashSet::<u32>::new();

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            if compressed_neighbors_in_mem.contains(&neighbor.id) {
                compressed_neighbors_in_mem_cnts += 1;
            }
            visited.insert(neighbor.id);

            if !dcc.update(neighbor.distance) {
                // Phase 1: full graph
                for &nn in self.graph.neighbors(neighbor.id as usize) {
                    let dist = self.pq_distance(nn, &pq_dists)?;
                    neighbor_pq.insert(Neighbor::new(nn, dist));
                }
            } else {
                // Phase 2: compressed neighbors only
                let compressed_nbrs = self.graph.compressed_neighbors(neighbor.id as usize);
                for &nn in compressed_nbrs {
                    let dist = self.pq_distance(nn, &pq_dists)?;
                    compressed_neighbors_in_mem.insert(nn);
                    if let Some(popped) = neighbor_pq.insert(Neighbor::new(nn, dist)) {
                        compressed_neighbors_in_mem.remove(&popped);
                    }
                }
                compressed_ndc += compressed_nbrs.len() as u32;
            }
        }

        // Rerank with exact L2 distance
        let mut result: Vec<(u32, f32)> = visited
            .iter()
            .map(|&id| {
                let dist = {
                    let qv = Vertex::new(query, 0);
                    let v = self.dataset.get_vertex(id).unwrap();
                    qv.compare(&v, Metric::L2)
                };
                (id, dist)
            })
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok((
            result.iter().map(|(id, _)| *id).take(k).collect(),
            visited.len() as u32,
            compressed_ndc,
            compressed_neighbors_in_mem_cnts,
        ))
    }

    // --- Profiling Search ---

    /// Search with detailed timing breakdown per phase.
    pub fn simulate_ssd_search_profiled(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
    ) -> ANNResult<(Vec<u32>, SearchProfile)> {
        let total_start = Instant::now();

        let t0 = Instant::now();
        let mut query_vec = query.to_vec();
        let (pq, _, _) = self.get_unwrapped_pq_component()?;
        pq.preprocess_query(&mut query_vec);
        let pq_dists = pq.populate_chunk_distances(&query_vec);
        let adc_table_us = t0.elapsed().as_secs_f64() * 1e6;

        let mut dcc = DistanceConvergenceChecker::new(window_size, epsilon);
        let entry = self.entry;
        let mut visited = HashSet::<u32>::new();

        let mut neighbor_pq = NeighborPriorityQueue::with_capacity(search_list_size);
        neighbor_pq.insert(Neighbor::new(entry, self.pq_distance(entry, &pq_dists)?));

        let mut phase1_us = 0.0f64;
        let mut phase2_us = 0.0f64;
        let mut phase1_iters: u32 = 0;
        let mut phase2_iters: u32 = 0;

        while neighbor_pq.has_notvisited_node() {
            let neighbor = neighbor_pq.closest_notvisited();
            visited.insert(neighbor.id);

            if !dcc.update(neighbor.distance) {
                let p1_start = Instant::now();
                phase1_iters += 1;
                for &nn in self.graph.neighbors(neighbor.id as usize) {
                    let dist = self.pq_distance(nn, &pq_dists)?;
                    neighbor_pq.insert(Neighbor::new(nn, dist));
                }
                phase1_us += p1_start.elapsed().as_secs_f64() * 1e6;
            } else {
                let p2_start = Instant::now();
                phase2_iters += 1;
                for &nn in self.graph.compressed_neighbors(neighbor.id as usize) {
                    let dist = self.pq_distance(nn, &pq_dists)?;
                    neighbor_pq.insert(Neighbor::new(nn, dist));
                }
                phase2_us += p2_start.elapsed().as_secs_f64() * 1e6;
            }
        }

        // Rerank
        let rerank_start = Instant::now();
        let mut result: Vec<(u32, f32)> = visited
            .iter()
            .map(|&id| {
                let dist = {
                    let qv = Vertex::new(query, 0);
                    let v = self.dataset.get_vertex(id).unwrap();
                    qv.compare(&v, Metric::L2)
                };
                (id, dist)
            })
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let rerank_us = rerank_start.elapsed().as_secs_f64() * 1e6;

        let total_us = total_start.elapsed().as_secs_f64() * 1e6;

        let profile = SearchProfile {
            total_us,
            adc_table_us,
            phase1_us,
            phase2_us,
            phase2_graph_read_us: 0.0,
            phase2_prefetch_us: 0.0,
            phase2_adc_us: 0.0,
            phase2_async_adc_us: 0.0,
            rerank_us,
            phase1_iters,
            phase2_iters,
            visited_count: visited.len() as u32,
            cache_hits: 0,
            async_batches: 0,
        };

        Ok((result.iter().map(|(id, _)| *id).take(k).collect(), profile))
    }
}
