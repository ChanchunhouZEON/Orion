/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! A/B compare the two `VisitedSet` implementations
//! ([`LinearProbeSet`] vs [`BucketedSet`]) on a synthetic
//! beam-search-shaped workload.
//!
//! Generates a controlled id stream that mimics the
//! `seen.insert(neighbour_id)` access pattern of the search hot path:
//! a population of `N_IDS` graph-id slots, drawn with a hit-rate
//! parameter that lets us sweep from "every insert is fresh"
//! (pre-convergence) to "most inserts are duplicates" (post-convergence).
//!
//! Usage:
//! ```bash
//! cargo run --release --bin hashset_profile -- [--L 128] [--n 5000000] [--hit 0.0,0.3,0.6]
//! ```

use orion::model::visited_set::{BucketedSet, LinearProbeSet, VisitedSet};
use std::time::Instant;

const DEFAULT_L: usize = 128;
const DEFAULT_R: usize = 100;
const DEFAULT_N_INSERTS: usize = 5_000_000;

/// Pseudo-random generator (splitmix64) — deterministic, no rand dep.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Build a synthetic id stream of length `n_inserts`. The id population
/// is `id_population` distinct u32 values; each insert draws from it
/// with `hit_rate` probability that the current draw collides with a
/// previously-drawn id (i.e., the visited-set already contains it).
fn build_workload(n_inserts: usize, id_population: usize, hit_rate: f64, seed: u64) -> Vec<u32> {
    let mut rng = SplitMix64::new(seed);

    // Pre-pick `id_population` distinct ids from a large random space so
    // there's no structural correlation with the hash multiplier.
    let mut ids: Vec<u32> = Vec::with_capacity(id_population);
    for _ in 0..id_population {
        ids.push(rng.next_u32() & 0x000F_FFFF); // ~1M id space
    }

    // Stream: with prob (1 - hit_rate) draw fresh (uniform from population),
    // with prob hit_rate draw from already-emitted ids (induce duplicates).
    let mut stream: Vec<u32> = Vec::with_capacity(n_inserts);
    for i in 0..n_inserts {
        let id = if i > 0 && rng.next_f64() < hit_rate {
            // Duplicate from earlier in the stream.
            let pick = rng.next_u32() as usize % i;
            stream[pick]
        } else {
            // Fresh draw from population.
            ids[rng.next_u32() as usize % id_population]
        };
        stream.push(id);
    }
    stream
}

/// Run a benchmark on a `VisitedSet` implementation. Returns
/// `(elapsed_ns, ns_per_insert, num_new_inserts)`.
fn bench<S: VisitedSet>(set: &mut S, workload: &[u32], clear_every: usize) -> (u128, f64, usize) {
    let n = workload.len();
    let t = Instant::now();
    let mut new_count = 0usize;
    for (i, &id) in workload.iter().enumerate() {
        if clear_every > 0 && i > 0 && i % clear_every == 0 {
            set.clear();
        }
        if set.insert(id) {
            new_count += 1;
        }
    }
    // Black-box the result so the optimizer can't elide.
    std::hint::black_box(new_count);
    let elapsed = t.elapsed().as_nanos();
    (elapsed, elapsed as f64 / n as f64, new_count)
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let mut args = std::env::args().peekable();
    while let Some(a) = args.next() {
        if a == format!("--{name}") {
            if let Some(v) = args.next() {
                if let Ok(parsed) = v.parse::<T>() {
                    return parsed;
                }
            }
        }
    }
    default
}

fn parse_hit_rates() -> Vec<f64> {
    let mut args = std::env::args().peekable();
    while let Some(a) = args.next() {
        if a == "--hit" {
            if let Some(v) = args.next() {
                return v
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
            }
        }
    }
    vec![0.0, 0.3, 0.6, 0.8]
}

fn main() {
    let l = parse_arg("L", DEFAULT_L);
    let r = parse_arg("R", DEFAULT_R);
    let n_inserts = parse_arg("n", DEFAULT_N_INSERTS);
    let hit_rates = parse_hit_rates();
    // Target peak load factor — drives `id_population` and `clear_every`.
    // The set has total capacity = next_pow2(2(L+1)R) slots; we size
    // the id stream so it fills the set to `--load` fraction before
    // clearing. Use 0.7+ to stress probe chains where the bucketed
    // SIMD scan should pay off vs linear's slot-by-slot walk.
    let load: f64 = parse_arg("load", 0.39_f64);
    let total_slots = (2 * (l + 1) * r).next_power_of_two();
    let id_population = ((total_slots as f64) * load).max(2048.0) as usize;
    let clear_every = id_population;

    println!(
        "─── HashsetSeen impl A/B ───\n  L={l}  R={r}  n_inserts={n_inserts}  id_population={id_population}  clear_every={clear_every}  target_load={load:.2}",
    );
    println!(
        "  Total slots per impl:  linear={}  bucketed={}",
        (2 * (l + 1) * r).next_power_of_two(),
        ((2 * (l + 1) * r).div_ceil(16)).next_power_of_two() * 16,
    );
    println!();

    println!(
        "  {:>8}  {:>14}  {:>14}  {:>14}  {:>10}",
        "hit_rate", "linear (ns/op)", "bucketed (ns/op)", "speedup", "new"
    );
    println!("  {}", "-".repeat(70));

    for &hit in &hit_rates {
        let workload = build_workload(n_inserts, id_population, hit, 0xC0FFEE);

        let mut linear = LinearProbeSet::new(l, r);
        let mut bucketed = BucketedSet::new(l, r);

        // Warm-up — fill caches, no timing.
        let warmup = &workload[..workload.len().min(50_000)];
        let _ = bench(&mut linear, warmup, clear_every);
        let _ = bench(&mut bucketed, warmup, clear_every);
        linear.clear();
        bucketed.clear();

        // Median of 5 timed runs.
        let mut linear_ns = Vec::new();
        let mut bucketed_ns = Vec::new();
        let mut linear_new = 0usize;
        let mut bucketed_new = 0usize;
        for _ in 0..5 {
            linear.clear();
            bucketed.clear();
            let (_, ns, new_l) = bench(&mut linear, &workload, clear_every);
            linear_ns.push(ns);
            linear_new = new_l;
            let (_, ns, new_b) = bench(&mut bucketed, &workload, clear_every);
            bucketed_ns.push(ns);
            bucketed_new = new_b;
        }
        linear_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bucketed_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let l_med = linear_ns[2];
        let b_med = bucketed_ns[2];

        // Sanity: both impls must agree on the new-insert count
        // modulo per-clear hash collisions (none — both are exact).
        assert_eq!(
            linear_new, bucketed_new,
            "impls diverged on new-insert count: linear={linear_new} bucketed={bucketed_new}"
        );

        println!(
            "  {:>8.2}  {:>14.2}  {:>14.2}  {:>13.2}x  {:>10}",
            hit,
            l_med,
            b_med,
            l_med / b_med,
            linear_new,
        );
    }
}
