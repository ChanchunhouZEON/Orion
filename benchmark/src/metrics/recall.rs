/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Calculate recall@K: fraction of ground truth neighbors found in the result set.
pub fn calculate_recall(results: &[u32], ground_truth: &[u32], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let found = results
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id))
        .count();
    found as f64 / k.min(ground_truth.len()) as f64
}

/// Calculate mean recall over multiple queries.
pub fn mean_recall(all_results: &[Vec<u32>], all_ground_truth: &[Vec<u32>], k: usize) -> f64 {
    if all_results.is_empty() {
        return 0.0;
    }
    let total: f64 = all_results
        .iter()
        .zip(all_ground_truth.iter())
        .map(|(r, gt)| calculate_recall(r, gt, k))
        .sum();
    total / all_results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_recall() {
        let results = vec![1, 2, 3, 4, 5];
        let gt = vec![1, 2, 3, 4, 5];
        assert!((calculate_recall(&results, &gt, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_zero_recall() {
        let results = vec![10, 20, 30];
        let gt = vec![1, 2, 3];
        assert!((calculate_recall(&results, &gt, 3) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_recall() {
        let results = vec![1, 2, 99];
        let gt = vec![1, 2, 3];
        assert!((calculate_recall(&results, &gt, 3) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_at_1() {
        let results = vec![5, 1, 2];
        let gt = vec![5, 1, 2];
        assert!((calculate_recall(&results, &gt, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_at_1_miss() {
        let results = vec![99, 1, 2];
        let gt = vec![5, 1, 2];
        assert!((calculate_recall(&results, &gt, 1) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_recall_perfect() {
        let results = vec![vec![1, 2], vec![3, 4]];
        let gt = vec![vec![1, 2], vec![3, 4]];
        assert!((mean_recall(&results, &gt, 2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_recall_empty() {
        let results: Vec<Vec<u32>> = vec![];
        let gt: Vec<Vec<u32>> = vec![];
        assert!((mean_recall(&results, &gt, 10) - 0.0).abs() < 1e-9);
    }
}
