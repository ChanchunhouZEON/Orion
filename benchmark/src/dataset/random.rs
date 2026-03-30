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

/// Brute-force k-NN using L2 distance.
pub fn brute_force_knn(query: &[f32], base: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, point)| {
            let dist = l2_distance(query, point);
            (i as u32, dist)
        })
        .collect();

    // Partial sort: only need top-k
    let k = k.min(dists.len());
    dists.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    dists.into_iter().map(|(id, _)| id).collect()
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
