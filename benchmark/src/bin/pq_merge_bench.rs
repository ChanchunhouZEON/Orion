/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Microbenchmark for the three `NeighborPriorityQueue` update paths:
//!   - per-insert (`pq.insert()` in a loop)
//!   - linear `batch_merge` (sort + set-union with mem::swap)
//!   - `batch_merge_gallop` (partition_point + bulk-copy)
//!
//! Sweeps PQ capacity `L` × incoming batch size `K` with random distances.
//! Per cell, reports mean time per call; annotates which path wins.
//!
//! Run with: `cargo run --release --bin pq_merge_bench`

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use orion::model::{Neighbor, NeighborPriorityQueue};
use std::time::Instant;

const SEED: u64 = 0xC0FFEE;
/// Per (L, K) cell: repeat `TRIALS` times, each trial does `OPS_PER_TRIAL`
/// flushes on a freshly-filled pq. Report median wall time.
const TRIALS: usize = 9;
const OPS_PER_TRIAL: usize = 2000;

fn rand_neighbors(rng: &mut StdRng, n: usize, start_id: u32) -> Vec<Neighbor> {
    (0..n)
        .map(|i| Neighbor::new(start_id + i as u32, rng.random_range(0.0_f32..1.0_f32)))
        .collect()
}

/// Fills a fresh pq to size `L` using per-insert so the layout is
/// representative of a real search-hot-path pq.
fn fresh_filled_pq(rng: &mut StdRng, l: usize) -> NeighborPriorityQueue {
    let mut pq = NeighborPriorityQueue::with_capacity(l);
    let init = rand_neighbors(rng, l * 2, 0);
    for nb in init {
        pq.insert(nb);
    }
    pq
}

/// Time per-insert: for each cand in cands, pq.insert(c).
fn bench_per_insert(l: usize, k: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut trial_ns: Vec<u128> = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        // Pre-generate all candidate batches so generation isn't in the hot loop.
        let mut batches: Vec<Vec<Neighbor>> = (0..OPS_PER_TRIAL)
            .map(|i| rand_neighbors(&mut rng, k, (l * 4 + i * k) as u32))
            .collect();
        let mut pq = fresh_filled_pq(&mut rng, l);
        let t = Instant::now();
        for batch in batches.drain(..) {
            for c in batch {
                pq.insert(c);
            }
        }
        trial_ns.push(t.elapsed().as_nanos());
    }
    trial_ns.sort_unstable();
    let med = trial_ns[TRIALS / 2] as f64;
    med / OPS_PER_TRIAL as f64
}

/// Time batch_merge (sorted cands already).
fn bench_batch_merge(l: usize, k: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut trial_ns: Vec<u128> = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let mut batches: Vec<Vec<Neighbor>> = (0..OPS_PER_TRIAL)
            .map(|i| {
                let mut v = rand_neighbors(&mut rng, k, (l * 4 + i * k) as u32);
                v.sort_unstable_by(|a, b| {
                    a.distance
                        .total_cmp(&b.distance)
                        .then_with(|| a.id.cmp(&b.id))
                });
                v
            })
            .collect();
        let mut pq = fresh_filled_pq(&mut rng, l);
        let mut scratch: Vec<Neighbor> = Vec::with_capacity(l + k + 1);
        let t = Instant::now();
        for batch in batches.drain(..) {
            pq.batch_merge(&batch, &mut scratch);
        }
        trial_ns.push(t.elapsed().as_nanos());
    }
    trial_ns.sort_unstable();
    let med = trial_ns[TRIALS / 2] as f64;
    med / OPS_PER_TRIAL as f64
}

/// Time batch_merge_gallop (sorted cands already).
fn bench_batch_merge_gallop(l: usize, k: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut trial_ns: Vec<u128> = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let mut batches: Vec<Vec<Neighbor>> = (0..OPS_PER_TRIAL)
            .map(|i| {
                let mut v = rand_neighbors(&mut rng, k, (l * 4 + i * k) as u32);
                v.sort_unstable_by(|a, b| {
                    a.distance
                        .total_cmp(&b.distance)
                        .then_with(|| a.id.cmp(&b.id))
                });
                v
            })
            .collect();
        let mut pq = fresh_filled_pq(&mut rng, l);
        let mut scratch: Vec<Neighbor> = Vec::with_capacity(l + k + 1);
        let t = Instant::now();
        for batch in batches.drain(..) {
            pq.batch_merge_gallop(&batch, &mut scratch);
        }
        trial_ns.push(t.elapsed().as_nanos());
    }
    trial_ns.sort_unstable();
    let med = trial_ns[TRIALS / 2] as f64;
    med / OPS_PER_TRIAL as f64
}

fn main() {
    let ls = [16usize, 32, 48, 64, 100, 128, 200, 256];
    let ks = [1usize, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

    println!(
        "{:>4}  {:>4}  {:>10}  {:>10}  {:>10}  {:>9}",
        "L", "K", "insert/ns", "merge/ns", "gallop/ns", "winner"
    );
    println!("{}", "─".repeat(58));
    for &l in &ls {
        for &k in &ks {
            if k > l * 2 {
                continue;
            }
            let t_ins = bench_per_insert(l, k);
            let t_mrg = bench_batch_merge(l, k);
            let t_glp = bench_batch_merge_gallop(l, k);
            let winner = {
                let mut best = ("insert", t_ins);
                if t_mrg < best.1 {
                    best = ("merge", t_mrg);
                }
                if t_glp < best.1 {
                    best = ("gallop", t_glp);
                }
                format!("{} ({:.0}ns)", best.0, best.1)
            };
            println!(
                "{:>4}  {:>4}  {:>10.0}  {:>10.0}  {:>10.0}  {:>9}",
                l, k, t_ins, t_mrg, t_glp, winner
            );
        }
        println!();
    }
}
