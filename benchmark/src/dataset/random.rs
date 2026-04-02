/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::Dataset;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;

/// Generate a random dataset with brute-force ground truth.
///
/// - `num_points`: number of base vectors
/// - `num_queries`: number of query vectors
/// - `dimension`: vector dimensionality
/// - `k`: depth of ground truth (top-k nearest neighbors)
/// - `seed`: RNG seed for reproducibility
pub fn generate_random_dataset(
    num_points: usize,
    num_queries: usize,
    dimension: usize,
    k: usize,
    seed: u64,
) -> Dataset {
    log::info!(
        "Generating random dataset: {} points, {} queries, dim={}, k={}",
        num_points,
        num_queries,
        dimension,
        k
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Generate base vectors
    let base: Vec<Vec<f32>> = (0..num_points)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.random_range(0f32..10f32))
                .collect()
        })
        .collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.random_range(0f32..10f32))
                .collect()
        })
        .collect();

    // Compute brute-force ground truth (parallel over queries)
    log::info!("Computing brute-force ground truth...");
    let ground_truth: Vec<Vec<u32>> = queries
        .par_iter()
        .map(|query| brute_force_knn(query, &base, k))
        .collect();

    log::info!("Random dataset generated successfully");

    Dataset {
        name: format!("random-{num_points}x{dimension}"),
        dimension,
        base,
        queries,
        ground_truth,
    }
}

/// Brute-force k-NN using L2 distance with bounded max-heap.
///
/// Uses a `BinaryHeap` of size k to track the k nearest neighbors in a
/// streaming fashion — O(n log k) time and O(k) memory instead of O(n).
pub fn brute_force_knn(query: &[f32], base: &[Vec<f32>], k: usize) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Max-heap by distance: the root is the farthest of the current top-k.
    // We store Reverse((OrderedFloat, id)) so BinaryHeap acts as a max-heap on distance.
    // Since f32 doesn't implement Ord, use u32 bits for total ordering.
    let k = k.min(base.len());
    let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1); // (dist_bits, id)

    for (i, point) in base.iter().enumerate() {
        let dist = l2_distance(query, point);
        let dist_bits = dist.to_bits(); // IEEE 754: bit ordering matches f32 ordering for non-negative

        if heap.len() < k {
            heap.push((dist_bits, i as u32));
        } else if let Some(&(max_bits, _)) = heap.peek() {
            if dist_bits < max_bits {
                heap.pop();
                heap.push((dist_bits, i as u32));
            }
        }
    }

    // Extract and sort by distance ascending.
    let mut result: Vec<(u32, u32)> = heap.into_vec();
    result.sort_unstable_by_key(|&(bits, _)| bits);
    result.into_iter().map(|(_, id)| id).collect()
}

#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}
