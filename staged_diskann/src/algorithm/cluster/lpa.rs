/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::algorithm::clustering_trait::{ClusteringResult, ClusteringStrategy};
use diskann::common::ANNResult;
use diskann::model::InMemoryGraph;
use ndarray::{ArcArray1, Array1};
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Label Propagation Algorithm for graph clustering.
pub struct LabelPropagationClustering {
    graph: Arc<InMemoryGraph>,
    num_nodes: usize,
    max_iterations: usize,
    max_cluster_size: usize,
}

impl LabelPropagationClustering {
    pub fn new(
        graph: Arc<InMemoryGraph>,
        num_nodes: usize,
        max_iterations: usize,
        max_cluster_size: usize,
    ) -> Self {
        Self {
            graph,
            num_nodes,
            max_iterations,
            max_cluster_size,
        }
    }

    /// Run LPA: each node adopts its most common neighbor label.
    fn run_lpa(&self) -> Vec<i32> {
        // Initialize: each node gets label = node_id
        let mut labels: Vec<i32> = (0..self.num_nodes as i32).collect();
        let mut order: Vec<usize> = (0..self.num_nodes).collect();
        let mut rng = rand::rng();

        for _round in 0..self.max_iterations {
            let mut changed = false;
            order.shuffle(&mut rng);

            for &node in &order {
                let neighbors = self.graph.to_neighbor_vec(node as u32).unwrap_or_default();
                if neighbors.is_empty() {
                    continue;
                }

                // Count neighbor labels
                let mut label_counts: HashMap<i32, usize> = HashMap::new();
                for &n in &neighbors {
                    let label = labels[n as usize];
                    *label_counts.entry(label).or_insert(0) += 1;
                }

                // Find most common label (tie-break by smallest label)
                let mut best_label = labels[node];
                let mut best_count = 0;
                for (&label, &count) in &label_counts {
                    if count > best_count || (count == best_count && label < best_label) {
                        best_count = count;
                        best_label = label;
                    }
                }

                if best_label != labels[node] {
                    labels[node] = best_label;
                    changed = true;
                }
            }

            if !changed {
                log::info!("LPA converged after {} iterations", _round + 1);
                break;
            }
        }

        labels
    }

    /// Remap labels to contiguous 0..k range.
    fn remap_labels(labels: &mut Vec<i32>) -> usize {
        let mut label_map: HashMap<i32, i32> = HashMap::new();
        let mut next_id = 0i32;

        for label in labels.iter_mut() {
            let new_id = *label_map.entry(*label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *label = new_id;
        }

        next_id as usize
    }

    /// Split clusters larger than max_cluster_size via BFS partitioning.
    fn split_oversized_clusters(&self, labels: &mut Vec<i32>) {
        let mut next_label = *labels.iter().max().unwrap_or(&0) + 1;

        loop {
            // Build cluster -> members map
            let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
            for (node, &label) in labels.iter().enumerate() {
                clusters.entry(label).or_default().push(node);
            }

            let mut any_split = false;

            for (_label, members) in &clusters {
                if members.len() <= self.max_cluster_size {
                    continue;
                }

                // BFS partition: split into chunks of max_cluster_size
                let member_set: HashSet<usize> = members.iter().copied().collect();
                let mut visited = HashSet::new();
                let mut current_chunk = Vec::new();

                for &start in members {
                    if visited.contains(&start) {
                        continue;
                    }

                    let mut queue = VecDeque::new();
                    queue.push_back(start);
                    visited.insert(start);

                    while let Some(node) = queue.pop_front() {
                        current_chunk.push(node);

                        if current_chunk.len() >= self.max_cluster_size {
                            // Assign new label to this chunk
                            for &n in &current_chunk {
                                labels[n] = next_label;
                            }
                            next_label += 1;
                            current_chunk.clear();
                            any_split = true;
                        }

                        let neighbors = self.graph.to_neighbor_vec(node as u32).unwrap_or_default();
                        for &n in &neighbors {
                            let n = n as usize;
                            if member_set.contains(&n) && visited.insert(n) {
                                queue.push_back(n);
                            }
                        }
                    }

                    if !current_chunk.is_empty() {
                        for &n in &current_chunk {
                            labels[n] = next_label;
                        }
                        next_label += 1;
                        current_chunk.clear();
                        any_split = true;
                    }
                }
            }

            if !any_split {
                break;
            }
        }
    }
}

impl ClusteringStrategy for LabelPropagationClustering {
    fn cluster(&mut self) -> ANNResult<ClusteringResult> {
        log::info!(
            "Running Label Propagation Algorithm (max_iter={}, max_cluster_size={})...",
            self.max_iterations,
            self.max_cluster_size
        );

        let mut labels = self.run_lpa();
        let num_clusters_before = Self::remap_labels(&mut labels);
        log::info!("LPA found {} clusters before split", num_clusters_before);

        self.split_oversized_clusters(&mut labels);
        let num_clusters_after = Self::remap_labels(&mut labels);
        log::info!(
            "After splitting oversized clusters: {} clusters",
            num_clusters_after
        );

        // Build storage_layout and point_affiliation
        let mut storage_layout: HashMap<u32, HashSet<u32>> = HashMap::new();
        for (node, &label) in labels.iter().enumerate() {
            storage_layout
                .entry(label as u32)
                .or_default()
                .insert(node as u32);
        }

        let point_affiliation: ArcArray1<i32> = Array1::from_vec(labels).to_shared();

        Ok(ClusteringResult {
            point_affiliation,
            storage_layout,
            centroids: HashMap::new(), // LPA does not compute centroids
        })
    }

    fn name(&self) -> &str {
        "LabelPropagation"
    }
}
