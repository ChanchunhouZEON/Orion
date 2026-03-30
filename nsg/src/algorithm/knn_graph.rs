use vector::{FullPrecisionDistance, Metric};

use crate::model::Neighbor;

/// Build a brute-force k-NN graph.
///
/// Returns `knn[i]` = sorted Vec of k nearest neighbors for point i.
pub fn build_knn_graph<T, const N: usize>(
    data: &[[T; N]],
    k: usize,
    metric: Metric,
) -> Vec<Vec<Neighbor>>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let n = data.len();
    let mut knn: Vec<Vec<Neighbor>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = <[T; N]>::distance_compare(&data[i], &data[j], metric);
            neighbors.push(Neighbor::new(j as u32, dist));
        }
        neighbors.sort();
        neighbors.truncate(k);
        knn.push(neighbors);

        if (i + 1) % 10000 == 0 {
            println!("  k-NN graph: {}/{} points", i + 1, n);
        }
    }

    knn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_correct_distances() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ];
        let knn = build_knn_graph(&data, 2, Metric::L2);

        // Node 0's nearest 2 should be node 1 (dist=1) and node 2 (dist=4)
        assert_eq!(knn[0].len(), 2);
        assert_eq!(knn[0][0].id, 1);
        assert_eq!(knn[0][1].id, 2);
    }

    #[test]
    fn test_knn_no_self_loops() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let knn = build_knn_graph(&data, 2, Metric::L2);

        for (i, neighbors) in knn.iter().enumerate() {
            for n in neighbors {
                assert_ne!(n.id, i as u32, "self-loop found at node {i}");
            }
        }
    }

    #[test]
    fn test_knn_sorted_by_distance() {
        let data: Vec<[f32; 4]> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let knn = build_knn_graph(&data, 3, Metric::L2);
        for neighbors in &knn {
            for w in neighbors.windows(2) {
                assert!(w[0].distance <= w[1].distance);
            }
        }
    }

    #[test]
    fn test_knn_k_truncation() {
        let data: Vec<[f32; 4]> = vec![
            [0.0; 4],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];
        let knn = build_knn_graph(&data, 2, Metric::L2);
        for neighbors in &knn {
            assert_eq!(neighbors.len(), 2);
        }
    }
}
