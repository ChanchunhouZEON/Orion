/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use hashbrown::HashSet;
use vector::{FullPrecisionDistance, Metric};

use crate::graph_mmap::MmapGraph;

/// A neighbor with distance, used for search results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmapNeighbor {
    pub id: u32,
    pub distance: f32,
}

impl Eq for MmapNeighbor {}

impl PartialOrd for MmapNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MmapNeighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Standalone greedy search on an mmap-backed DiskANN graph.
///
/// Reads neighbors and vectors directly from the memory-mapped file.
/// No in-memory graph reconstruction is needed.
///
/// # Arguments
/// * `graph` - Memory-mapped graph file (must contain vectors, dimension > 0)
/// * `query` - Query vector (f32 array)
/// * `start` - Entry point node ID
/// * `search_list_size` - Size of the candidate list (L parameter)
/// * `k` - Number of nearest neighbors to return
/// * `metric` - Distance metric (L2 or Cosine)
pub fn mmap_greedy_search<const N: usize>(
    graph: &MmapGraph,
    query: &[f32; N],
    start: u32,
    search_list_size: usize,
    k: usize,
    metric: Metric,
) -> Vec<MmapNeighbor>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let get_vector = |node_id: u32| -> [f32; N] {
        let vec_slice = graph.vector_f32(node_id);
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(vec_slice);
        arr
    };
    greedy_search_core(
        query,
        start,
        search_list_size,
        k,
        metric,
        |id| graph.neighbors(id).to_vec(),
        get_vector,
    )
}

/// Greedy search reading neighbors from mmap, distances from in-memory data.
///
/// Use this when the mmap graph file only contains adjacency data (dimension=0)
/// and vector data is stored separately in memory.
pub fn mmap_greedy_search_with_data<const N: usize>(
    graph: &MmapGraph,
    data: &[[f32; N]],
    query: &[f32; N],
    start: u32,
    search_list_size: usize,
    k: usize,
    metric: Metric,
) -> Vec<MmapNeighbor>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    greedy_search_core(
        query,
        start,
        search_list_size,
        k,
        metric,
        |id| graph.neighbors(id).to_vec(),
        |id| data[id as usize],
    )
}

/// Core greedy search generic over neighbor and vector access.
fn greedy_search_core<const N: usize>(
    query: &[f32; N],
    start: u32,
    search_list_size: usize,
    k: usize,
    metric: Metric,
    get_neighbors: impl Fn(u32) -> Vec<u32>,
    get_vector: impl Fn(u32) -> [f32; N],
) -> Vec<MmapNeighbor>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let l = search_list_size.max(k);

    let mut visited = HashSet::<u32>::new();
    let mut candidates: BinaryHeap<Reverse<MmapNeighbor>> = BinaryHeap::new();
    let mut results: BinaryHeap<MmapNeighbor> = BinaryHeap::new();

    let start_arr = get_vector(start);
    let start_dist = <[f32; N]>::distance_compare(query, &start_arr, metric);

    visited.insert(start);
    candidates.push(Reverse(MmapNeighbor {
        id: start,
        distance: start_dist,
    }));
    results.push(MmapNeighbor {
        id: start,
        distance: start_dist,
    });

    while let Some(Reverse(current)) = candidates.pop() {
        let farthest_result = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
        if current.distance > farthest_result {
            break;
        }

        let neighbors = get_neighbors(current.id);

        for neighbor_id in neighbors {
            if visited.insert(neighbor_id) {
                let arr = get_vector(neighbor_id);
                let dist = <[f32; N]>::distance_compare(query, &arr, metric);

                let farthest_result = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                if results.len() < l || dist < farthest_result {
                    candidates.push(Reverse(MmapNeighbor {
                        id: neighbor_id,
                        distance: dist,
                    }));
                    results.push(MmapNeighbor {
                        id: neighbor_id,
                        distance: dist,
                    });

                    if results.len() > l {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut result_vec: Vec<MmapNeighbor> = results.drain().collect();
    result_vec.sort();
    result_vec.truncate(k);
    result_vec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_mmap::{AlgorithmId, GraphHeader, GraphWriter};

    fn create_test_graph<const N: usize>(
        data: &[[f32; N]],
        adjacency: &[Vec<u32>],
    ) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("anns_mmap_search_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_search_graph.anns");

        let header = GraphHeader {
            num_nodes: data.len() as u32,
            max_degree: adjacency.iter().map(|a| a.len()).max().unwrap_or(0) as u32,
            dimension: N as u32,
            metric: 0, // L2
            algorithm: AlgorithmId::DiskAnn,
        };

        let writer = GraphWriter::new(header);
        writer
            .write(
                &path,
                |node_id| {
                    let neighbors = &adjacency[node_id as usize];
                    (neighbors.len() as u32, neighbors.clone())
                },
                |node_id| data[node_id as usize].to_vec(),
            )
            .unwrap();

        path
    }

    #[test]
    fn test_mmap_greedy_search_finds_nearest() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];
        let adjacency = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];

        let path = create_test_graph(&data, &adjacency);
        let graph = MmapGraph::open(&path, 0).unwrap();

        let query = [2.0, 0.0, 0.0, 0.0];
        let results = mmap_greedy_search::<4>(&graph, &query, 0, 5, 3, Metric::L2);

        assert!(!results.is_empty());
        assert_eq!(results[0].id, 2); // exact match

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_mmap_greedy_search_results_sorted() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];
        let adjacency = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];

        let path = create_test_graph(&data, &adjacency);
        let graph = MmapGraph::open(&path, 0).unwrap();

        let query = [2.5, 0.0, 0.0, 0.0];
        let results = mmap_greedy_search::<4>(&graph, &query, 0, 5, 5, Metric::L2);

        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
