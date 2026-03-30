use std::collections::{HashSet, VecDeque};

use crate::common::HNSWResult;
use crate::model::HNSWGraphAccess;

/// BFS from entry point for `max_hops` hops, returning all visited node IDs.
///
/// This is used to warm the OS page cache before search by touching
/// the neighborhood around the entry point.
pub fn bfs_neighborhood<G: HNSWGraphAccess>(
    graph: &G,
    entry: u32,
    layer: usize,
    max_hops: usize,
) -> HNSWResult<HashSet<u32>> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(entry);
    queue.push_back((entry, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }

        let neighbors = graph.get_neighbors(node, layer)?;
        for &neighbor in &neighbors {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    Ok(visited)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HNSWGraph;

    #[test]
    fn test_bfs_neighborhood_single_hop() {
        let graph = HNSWGraph::new(5, 4, 4);
        graph.set_neighbors(0, 0, vec![1, 2]).unwrap();
        graph.set_neighbors(1, 0, vec![0, 3]).unwrap();
        graph.set_neighbors(2, 0, vec![0, 4]).unwrap();
        graph.set_neighbors(3, 0, vec![1]).unwrap();
        graph.set_neighbors(4, 0, vec![2]).unwrap();

        let result = bfs_neighborhood(&graph, 0, 0, 1).unwrap();
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(!result.contains(&3));
        assert!(!result.contains(&4));
    }

    #[test]
    fn test_bfs_neighborhood_two_hops() {
        let graph = HNSWGraph::new(5, 4, 4);
        graph.set_neighbors(0, 0, vec![1, 2]).unwrap();
        graph.set_neighbors(1, 0, vec![0, 3]).unwrap();
        graph.set_neighbors(2, 0, vec![0, 4]).unwrap();
        graph.set_neighbors(3, 0, vec![1]).unwrap();
        graph.set_neighbors(4, 0, vec![2]).unwrap();

        let result = bfs_neighborhood(&graph, 0, 0, 2).unwrap();
        assert_eq!(result.len(), 5); // all nodes reachable in 2 hops
    }

    #[test]
    fn test_bfs_zero_hops() {
        let graph = HNSWGraph::new(3, 4, 4);
        graph.set_neighbors(0, 0, vec![1, 2]).unwrap();

        let result = bfs_neighborhood(&graph, 0, 0, 0).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }
}
