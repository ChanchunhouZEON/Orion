use std::cmp::Reverse;
use std::collections::BinaryHeap;

use hashbrown::HashSet;
use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::common::HNSWResult;
use crate::model::{HNSWGraphAccess, HNSWScratch, Neighbor};

/// Search within a single layer of the HNSW graph.
///
/// Returns up to `ef` nearest neighbors at the given layer.
///
/// Uses a two-heap approach:
/// - `candidates` (min-heap): nodes to explore, ordered by distance ascending
/// - `results` (max-heap): current best results, ordered by distance descending
///
/// If `scratch` is provided, uses pre-allocated buffers to avoid per-query allocations.
pub fn search_layer<T, const N: usize, G: HNSWGraphAccess, V: VectorStorage<T, N> + ?Sized>(
    query: &[T; N],
    entry_point: u32,
    ef: usize,
    layer: usize,
    graph: &G,
    data: &V,
    metric: Metric,
) -> HNSWResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    // Allocate fresh buffers (non-scratch path)
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
    let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

    search_layer_inner(
        query,
        entry_point,
        ef,
        layer,
        graph,
        data,
        metric,
        &mut visited,
        &mut candidates,
        &mut results,
    )
}

/// Search within a single layer using pre-allocated scratch buffers.
pub fn search_layer_with_scratch<
    T,
    const N: usize,
    G: HNSWGraphAccess,
    V: VectorStorage<T, N> + ?Sized,
>(
    query: &[T; N],
    entry_point: u32,
    ef: usize,
    layer: usize,
    graph: &G,
    data: &V,
    metric: Metric,
    scratch: &mut HNSWScratch,
) -> HNSWResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    search_layer_inner(
        query,
        entry_point,
        ef,
        layer,
        graph,
        data,
        metric,
        &mut scratch.visited,
        &mut scratch.candidates,
        &mut scratch.results,
    )
}

fn search_layer_inner<T, const N: usize, G: HNSWGraphAccess, V: VectorStorage<T, N> + ?Sized>(
    query: &[T; N],
    entry_point: u32,
    ef: usize,
    layer: usize,
    graph: &G,
    data: &V,
    metric: Metric,
    visited: &mut HashSet<u32>,
    candidates: &mut BinaryHeap<Reverse<Neighbor>>,
    results: &mut BinaryHeap<Neighbor>,
) -> HNSWResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let entry_dist = distance::<T, N>(query, &data.get_vector(entry_point), metric);

    visited.insert(entry_point);
    candidates.push(Reverse(Neighbor::new(entry_point, entry_dist)));
    results.push(Neighbor::new(entry_point, entry_dist));

    while let Some(Reverse(current)) = candidates.pop() {
        // If the closest candidate is farther than the farthest result, stop.
        let farthest_result = results.peek().unwrap().distance;
        if current.distance > farthest_result {
            break;
        }

        let neighbors = graph.get_neighbors(current.id, layer)?;

        for &neighbor_id in &neighbors {
            if visited.insert(neighbor_id) {
                let dist = distance::<T, N>(query, &data.get_vector(neighbor_id), metric);
                let farthest_result = results.peek().unwrap().distance;

                if results.len() < ef || dist < farthest_result {
                    candidates.push(Reverse(Neighbor::new(neighbor_id, dist)));
                    results.push(Neighbor::new(neighbor_id, dist));

                    if results.len() > ef {
                        results.pop(); // Remove farthest
                    }
                }
            }
        }
    }

    // Convert results heap to sorted vector (closest first).
    let mut result_vec: Vec<Neighbor> = results.drain().collect();
    result_vec.sort();
    Ok(result_vec)
}

/// Greedily search from entry_point through upper layers (ef=1),
/// returning the closest node at the target layer.
pub fn search_upper_layers<T, const N: usize, G: HNSWGraphAccess, V: VectorStorage<T, N> + ?Sized>(
    query: &[T; N],
    entry_point: u32,
    top_layer: usize,
    target_layer: usize,
    graph: &G,
    data: &V,
    metric: Metric,
) -> HNSWResult<u32>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let mut current = entry_point;
    let mut current_dist = distance::<T, N>(query, &data.get_vector(current), metric);

    for layer in (target_layer..=top_layer).rev() {
        let mut changed = true;
        while changed {
            changed = false;
            let neighbors = graph.get_neighbors(current, layer)?;

            for &neighbor_id in &neighbors {
                let dist = distance::<T, N>(query, &data.get_vector(neighbor_id), metric);
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    changed = true;
                }
            }
        }
    }

    Ok(current)
}

/// Compute distance between two vectors using the SIMD distance trait.
#[inline]
fn distance<T, const N: usize>(a: &[T; N], b: &[T; N], metric: Metric) -> f32
where
    T: Default + Copy + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    <[T; N]>::distance_compare(a, b, metric)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HNSWGraph;

    fn make_small_graph_and_data() -> (HNSWGraph, Vec<[f32; 4]>) {
        // 5 nodes in 4D, connected in a chain: 0-1-2-3-4
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];

        let mut graph = HNSWGraph::new(5, 4, 8);
        graph.set_num_nodes(5);
        // Chain connections at layer 0
        graph.set_neighbors(0, 0, vec![1]).unwrap();
        graph.set_neighbors(1, 0, vec![0, 2]).unwrap();
        graph.set_neighbors(2, 0, vec![1, 3]).unwrap();
        graph.set_neighbors(3, 0, vec![2, 4]).unwrap();
        graph.set_neighbors(4, 0, vec![3]).unwrap();

        (graph, data)
    }

    #[test]
    fn test_search_layer_finds_nearest() {
        let (graph, data) = make_small_graph_and_data();
        let query = [0.5, 0.0, 0.0, 0.0];
        let results = search_layer(&query, 0, 5, 0, &graph, &data, Metric::L2).unwrap();
        // Closest should be node 0 (dist=0.25) then node 1 (dist=0.25)
        assert!(!results.is_empty());
        assert!(results[0].id == 0 || results[0].id == 1);
    }

    #[test]
    fn test_search_layer_ef_1() {
        let (graph, data) = make_small_graph_and_data();
        let query = [0.0, 0.0, 0.0, 0.0];
        let results = search_layer(&query, 0, 1, 0, &graph, &data, Metric::L2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_search_layer_with_scratch() {
        let (graph, data) = make_small_graph_and_data();
        let query = [2.0, 0.0, 0.0, 0.0];
        let mut scratch = HNSWScratch::new(10, 10);
        let results =
            search_layer_with_scratch(&query, 0, 3, 0, &graph, &data, Metric::L2, &mut scratch)
                .unwrap();
        assert_eq!(results[0].id, 2); // exact match
    }

    #[test]
    fn test_search_upper_layers() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ];
        let mut graph = HNSWGraph::new(3, 4, 8);
        graph.set_num_nodes(3);
        graph.ensure_layers(1, 3);
        // Layer 0: all connected
        graph.set_neighbors(0, 0, vec![1, 2]).unwrap();
        graph.set_neighbors(1, 0, vec![0, 2]).unwrap();
        graph.set_neighbors(2, 0, vec![0, 1]).unwrap();
        // Layer 1: 0 and 2 connected
        graph.set_neighbors(0, 1, vec![2]).unwrap();
        graph.set_neighbors(2, 1, vec![0]).unwrap();
        graph.max_level = 1;

        let query = [0.5, 0.0, 0.0, 0.0];
        let result = search_upper_layers(&query, 2, 1, 1, &graph, &data, Metric::L2).unwrap();
        assert_eq!(result, 0); // node 0 is closer to query than node 2 at layer 1
    }

    #[test]
    fn test_search_results_sorted() {
        let (graph, data) = make_small_graph_and_data();
        let query = [2.5, 0.0, 0.0, 0.0];
        let results = search_layer(&query, 0, 5, 0, &graph, &data, Metric::L2).unwrap();
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }
}
