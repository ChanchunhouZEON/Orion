use std::cmp::Reverse;
use std::collections::BinaryHeap;

use hashbrown::HashSet;
use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::common::NSGResult;
use crate::model::{NSGGraphAccess, NSGScratch, Neighbor};

/// Greedy best-first search from a starting node.
///
/// Returns up to `l` nearest neighbors sorted by distance.
pub fn greedy_search<T, const N: usize, G: NSGGraphAccess, V: VectorStorage<T, N> + ?Sized>(
    query: &[T; N],
    start: u32,
    l: usize,
    graph: &G,
    data: &V,
    metric: Metric,
) -> NSGResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let mut visited: HashSet<u32> = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<Neighbor>> = BinaryHeap::new();
    let mut results: BinaryHeap<Neighbor> = BinaryHeap::new();

    greedy_search_inner(
        query,
        start,
        l,
        graph,
        data,
        metric,
        &mut visited,
        &mut candidates,
        &mut results,
    )
}

/// Greedy search using pre-allocated scratch buffers.
pub fn greedy_search_with_scratch<
    T,
    const N: usize,
    G: NSGGraphAccess,
    V: VectorStorage<T, N> + ?Sized,
>(
    query: &[T; N],
    start: u32,
    l: usize,
    graph: &G,
    data: &V,
    metric: Metric,
    scratch: &mut NSGScratch,
) -> NSGResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    greedy_search_inner(
        query,
        start,
        l,
        graph,
        data,
        metric,
        &mut scratch.visited,
        &mut scratch.candidates,
        &mut scratch.results,
    )
}

fn greedy_search_inner<T, const N: usize, G: NSGGraphAccess, V: VectorStorage<T, N> + ?Sized>(
    query: &[T; N],
    start: u32,
    l: usize,
    graph: &G,
    data: &V,
    metric: Metric,
    visited: &mut HashSet<u32>,
    candidates: &mut BinaryHeap<Reverse<Neighbor>>,
    results: &mut BinaryHeap<Neighbor>,
) -> NSGResult<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let start_dist = <[T; N]>::distance_compare(query, &data.get_vector(start), metric);

    visited.insert(start);
    candidates.push(Reverse(Neighbor::new(start, start_dist)));
    results.push(Neighbor::new(start, start_dist));

    while let Some(Reverse(current)) = candidates.pop() {
        let farthest_result = results.peek().unwrap().distance;
        if current.distance > farthest_result {
            break;
        }

        let neighbors = graph.get_neighbors(current.id)?;

        for &neighbor_id in &neighbors {
            if visited.insert(neighbor_id) {
                let dist = <[T; N]>::distance_compare(query, &data.get_vector(neighbor_id), metric);
                let farthest_result = results.peek().unwrap().distance;

                if results.len() < l || dist < farthest_result {
                    candidates.push(Reverse(Neighbor::new(neighbor_id, dist)));
                    results.push(Neighbor::new(neighbor_id, dist));

                    if results.len() > l {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut result_vec: Vec<Neighbor> = results.drain().collect();
    result_vec.sort();
    Ok(result_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NSGGraph;

    fn make_chain_graph() -> (NSGGraph, Vec<[f32; 4]>) {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];
        let graph = NSGGraph::new(5, 4);
        graph.set_neighbors(0, vec![1]).unwrap();
        graph.set_neighbors(1, vec![0, 2]).unwrap();
        graph.set_neighbors(2, vec![1, 3]).unwrap();
        graph.set_neighbors(3, vec![2, 4]).unwrap();
        graph.set_neighbors(4, vec![3]).unwrap();
        (graph, data)
    }

    #[test]
    fn test_greedy_search_finds_nearest() {
        let (graph, data) = make_chain_graph();
        let query = [2.0, 0.0, 0.0, 0.0];
        let results = greedy_search(&query, 0, 5, &graph, &data, Metric::L2).unwrap();
        assert_eq!(results[0].id, 2); // exact match
    }

    #[test]
    fn test_greedy_search_results_ordered() {
        let (graph, data) = make_chain_graph();
        let query = [2.5, 0.0, 0.0, 0.0];
        let results = greedy_search(&query, 0, 5, &graph, &data, Metric::L2).unwrap();
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_greedy_search_with_scratch() {
        let (graph, data) = make_chain_graph();
        let query = [1.0, 0.0, 0.0, 0.0];
        let mut scratch = NSGScratch::new(10, 10);
        let results =
            greedy_search_with_scratch(&query, 0, 3, &graph, &data, Metric::L2, &mut scratch)
                .unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_greedy_search_l_1() {
        let (graph, data) = make_chain_graph();
        let query = [0.0, 0.0, 0.0, 0.0];
        let results = greedy_search(&query, 0, 1, &graph, &data, Metric::L2).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }
}
