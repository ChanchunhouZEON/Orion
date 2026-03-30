use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::model::Neighbor;

/// NSG edge selection: select up to R neighbors from the candidate pool
/// ensuring the monotonic search path property.
///
/// For each candidate in order of distance, add it if it is not "occluded"
/// by a closer already-selected neighbor (similar to DiskANN's robust_prune).
pub fn select_edges<T, const N: usize, V: VectorStorage<T, N> + ?Sized>(
    candidates: &[Neighbor],
    r: usize,
    data: &V,
    metric: Metric,
) -> Vec<u32>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort();

    let mut selected: Vec<Neighbor> = Vec::with_capacity(r);

    for &candidate in &sorted {
        if selected.len() >= r {
            break;
        }

        // Check NSG monotonic path constraint:
        // Accept candidate p if for all already-selected s,
        // dist(node, p) < dist(s, p).
        let occluded = selected.iter().any(|s| {
            let dist_sp = <[T; N]>::distance_compare(
                &data.get_vector(s.id),
                &data.get_vector(candidate.id),
                metric,
            );
            dist_sp <= candidate.distance
        });

        if !occluded {
            selected.push(candidate);
        }
    }

    // Fill remaining slots with closest unselected candidates.
    if selected.len() < r {
        for &candidate in &sorted {
            if selected.len() >= r {
                break;
            }
            if !selected.iter().any(|s| s.id == candidate.id) {
                selected.push(candidate);
            }
        }
    }

    selected.iter().map(|n| n.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_edges_r_limit() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ];
        let candidates = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 4.0),
            Neighbor::new(3, 9.0),
        ];
        let selected = select_edges(&candidates, 2, &data, Metric::L2);
        assert!(selected.len() <= 2);
    }

    #[test]
    fn test_select_edges_empty() {
        let data: Vec<[f32; 4]> = vec![[0.0; 4]];
        let selected = select_edges(&[], 5, &data, Metric::L2);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_select_edges_fewer_than_r() {
        let data: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let candidates = vec![Neighbor::new(1, 1.0)];
        let selected = select_edges(&candidates, 5, &data, Metric::L2);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 1);
    }

    #[test]
    fn test_select_edges_diversity() {
        // Two points close together and one far away in a different direction
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0], // reference node
            [1.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0, 0.0], // close to node 1
            [0.0, 5.0, 0.0, 0.0], // different direction
        ];
        let candidates = vec![
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 1.21),
            Neighbor::new(3, 25.0),
        ];
        let selected = select_edges(&candidates, 2, &data, Metric::L2);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1));
        // Should prefer diversity: node 3 is in different direction
        assert!(selected.contains(&3));
    }
}
