use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::common::HNSWResult;
use crate::model::{HNSWGraph, Neighbor};

use super::search::{search_layer, search_upper_layers};

/// Select the M closest neighbors from candidates using the simple heuristic.
pub fn select_neighbors_simple(candidates: &[Neighbor], m: usize) -> Vec<u32> {
    let mut sorted = candidates.to_vec();
    sorted.sort();
    sorted.iter().take(m).map(|n| n.id).collect()
}

/// Select neighbors using the heuristic that promotes diversity.
/// Similar to DiskANN's robust_prune: for each candidate in order of distance,
/// add it only if it is not "occluded" by a closer already-selected neighbor.
pub fn select_neighbors_heuristic<T, const N: usize, V: VectorStorage<T, N> + ?Sized>(
    candidates: &[Neighbor],
    m: usize,
    data: &V,
    metric: Metric,
) -> Vec<u32>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    if candidates.len() <= m {
        return candidates.iter().map(|n| n.id).collect();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort();

    let mut selected: Vec<Neighbor> = Vec::with_capacity(m);

    for &candidate in &sorted {
        if selected.len() >= m {
            break;
        }

        // Check if this candidate is "occluded" by any already-selected neighbor.
        let occluded = selected.iter().any(|s| {
            let dist_between = <[T; N]>::distance_compare(
                &data.get_vector(s.id),
                &data.get_vector(candidate.id),
                metric,
            );
            dist_between < candidate.distance
        });

        if !occluded {
            selected.push(candidate);
        }
    }

    // If we didn't get enough from the heuristic, fill with remaining closest.
    if selected.len() < m {
        for &candidate in &sorted {
            if selected.len() >= m {
                break;
            }
            if !selected.iter().any(|s| s.id == candidate.id) {
                selected.push(candidate);
            }
        }
    }

    selected.iter().map(|n| n.id).collect()
}

/// Insert a node into the HNSW graph.
///
/// 1. Assign a random level.
/// 2. Search from entry point down to the node's level to find nearest neighbors.
/// 3. At each layer from node's level down to 0, connect the node to M nearest neighbors.
/// 4. If the node's level is higher than the current max level, update the entry point.
pub fn insert_node<T, const N: usize, V: VectorStorage<T, N> + ?Sized>(
    node_id: u32,
    level: usize,
    graph: &mut HNSWGraph,
    data: &V,
    metric: Metric,
    ef_construction: usize,
    use_heuristic: bool,
) -> HNSWResult<()>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let query = data.get_vector(node_id);

    graph.set_node_level(node_id, level);

    if graph.num_nodes() == 0 {
        // First node: just set as entry point.
        graph.entry_point = node_id;
        graph.max_level = level;
        graph.ensure_layers(level, data.num_vectors());
        graph.set_num_nodes(1);
        return Ok(());
    }

    // Ensure graph has enough layers.
    let target_level = level.max(graph.max_level);
    graph.ensure_layers(target_level, data.num_vectors());

    let mut entry_point = graph.entry_point;

    // Greedy search through upper layers (above node's level).
    if graph.max_level > level {
        entry_point = search_upper_layers(
            &query,
            entry_point,
            graph.max_level,
            level + 1,
            graph,
            data,
            metric,
        )?;
    }

    // At each layer from min(level, max_level) down to 0:
    let start_layer = level.min(graph.max_level);
    for lc in (0..=start_layer).rev() {
        // Search for ef_construction nearest neighbors at this layer.
        let candidates = search_layer(
            &query,
            entry_point,
            ef_construction,
            lc,
            graph,
            data,
            metric,
        )?;

        // Select M neighbors.
        let m = graph.max_connections(lc);
        let selected = if use_heuristic {
            select_neighbors_heuristic(&candidates, m, data, metric)
        } else {
            select_neighbors_simple(&candidates, m)
        };

        // Connect the new node to selected neighbors.
        graph.set_neighbors(node_id, lc, selected.clone())?;

        // Add bidirectional connections.
        for &neighbor_id in &selected {
            graph.add_neighbor(neighbor_id, node_id, lc)?;

            // Prune neighbor's connections if they exceed the limit.
            let neighbor_connections = graph.get_neighbors(neighbor_id, lc)?;
            if neighbor_connections.len() > m {
                // Re-select M neighbors for this node.
                let neighbor_candidates: Vec<Neighbor> = neighbor_connections
                    .iter()
                    .map(|&nid| {
                        let dist = <[T; N]>::distance_compare(
                            &data.get_vector(neighbor_id),
                            &data.get_vector(nid),
                            metric,
                        );
                        Neighbor::new(nid, dist)
                    })
                    .collect();

                let pruned = if use_heuristic {
                    select_neighbors_heuristic(&neighbor_candidates, m, data, metric)
                } else {
                    select_neighbors_simple(&neighbor_candidates, m)
                };
                graph.set_neighbors(neighbor_id, lc, pruned)?;
            }
        }

        // Use the closest result as entry point for the next layer down.
        if !candidates.is_empty() {
            entry_point = candidates[0].id;
        }
    }

    // Update entry point if this node has a higher level.
    if level > graph.max_level {
        graph.entry_point = node_id;
        graph.max_level = level;
    }

    graph.set_num_nodes(graph.num_nodes() + 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_first_node() {
        let data: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 0.0]];
        let mut graph = HNSWGraph::new(1, 4, 8);

        insert_node(0, 0, &mut graph, &data, Metric::L2, 10, false).unwrap();
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.entry_point, 0);
    }

    #[test]
    fn test_insert_updates_entry_point_on_higher_level() {
        let data: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let mut graph = HNSWGraph::new(2, 4, 8);

        insert_node(0, 0, &mut graph, &data, Metric::L2, 10, false).unwrap();
        assert_eq!(graph.entry_point, 0);
        assert_eq!(graph.max_level, 0);

        insert_node(1, 2, &mut graph, &data, Metric::L2, 10, false).unwrap();
        assert_eq!(graph.entry_point, 1); // higher level => new entry point
        assert_eq!(graph.max_level, 2);
    }

    #[test]
    fn test_insert_bidirectional_connections() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let mut graph = HNSWGraph::new(3, 4, 8);

        for i in 0..3 {
            insert_node(i, 0, &mut graph, &data, Metric::L2, 10, false).unwrap();
        }

        // All nodes should have connections at layer 0
        for i in 0..3u32 {
            let neighbors = graph.get_neighbors(i, 0).unwrap();
            assert!(!neighbors.is_empty(), "node {i} should have neighbors");
        }
    }

    #[test]
    fn test_select_neighbors_simple() {
        let candidates = vec![
            Neighbor::new(0, 3.0),
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 2.0),
        ];
        let selected = select_neighbors_simple(&candidates, 2);
        assert_eq!(selected, vec![1, 2]); // closest 2
    }

    #[test]
    fn test_select_neighbors_heuristic_diversity() {
        // 4 data points in 4D, two of them are very close (potential occlusion)
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0], // query reference
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0, 0.0], // close to node 1
            [0.0, 5.0, 0.0, 0.0], // far but diverse direction
        ];
        let candidates = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 1.21),
            Neighbor::new(3, 25.0),
        ];
        let selected = select_neighbors_heuristic(&candidates, 2, &data, Metric::L2);
        // Heuristic should prefer diversity: node 1 and node 3 (diverse direction)
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1));
        assert!(selected.contains(&3)); // diverse, not occluded by node 1
    }
}
