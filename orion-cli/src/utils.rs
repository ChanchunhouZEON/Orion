/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Shared benchmark-harness utilities.
//!
//! Consumed by several bins via `#[path = "../utils.rs"] mod utils`
//! (e.g. `orion.rs`, `orion.rs`). The benchmark crate
//! itself doesn't reach every export, so we silence the dead-code
//! warnings at the module level rather than tagging each helper.

#![allow(dead_code)]

use std::hint::black_box;

/// One registry of the const-generic dimensions compiled into the sweep driver.
/// Padding is a data transformation, not a dimension alias.
#[macro_export]
macro_rules! with_supported_dimension {
    ($dimension:expr, |$d:ident| $body:expr) => {
        match $dimension {
            32 => { const $d: usize = 32; $body }
            100 => { const $d: usize = 100; $body }
            128 => { const $d: usize = 128; $body }
            768 => { const $d: usize = 768; $body }
            784 => { const $d: usize = 784; $body }
            960 => { const $d: usize = 960; $body }
            1536 => { const $d: usize = 1536; $body }
            other => Err(format!("unsupported physical dimension {other}; explicitly pad data or add a compiled specialization")),
        }
    };
}

/// Validate without silently clamping, reordering, or repeating experiments.
pub fn validate_search_list_sizes(k: usize, values: &[usize]) -> Result<(), String> {
    if k == 0 {
        return Err("k must be positive".into());
    }
    if values.is_empty() {
        return Err("search-list-sizes must not be empty".into());
    }
    if values.iter().any(|&l| l < k) {
        return Err(format!("every search L must be >= k={k}; got {values:?}"));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("search-list-sizes must be strictly increasing (no duplicates)".into());
    }
    Ok(())
}

pub fn validate_ground_truth(gt: &[Vec<u32>], queries: usize, k: usize) -> Result<(), String> {
    if k == 0 || queries == 0 || gt.len() < queries || gt[..queries].iter().any(|row| row.len() < k)
    {
        return Err(format!(
            "ground truth must contain {queries} query rows with at least k={k} IDs each"
        ));
    }
    Ok(())
}

/// Preserve published top-10 paths; do not overwrite them with other k runs.
pub fn result_path(prefix: &str, dataset: &str, k: usize) -> String {
    let suffix = if k == 10 {
        String::new()
    } else {
        format!("_k{k}")
    };
    format!("visualizations/{prefix}_{dataset}{suffix}.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_ground_truth_depth_and_query_count() {
        let gt = vec![vec![0; 100]; 2];
        assert!(validate_ground_truth(&gt, 2, 100).is_ok());
        for (queries, k) in [(3, 100), (2, 101), (0, 10), (2, 0)] {
            assert!(validate_ground_truth(&gt, queries, k).is_err());
        }
        assert!(validate_ground_truth(&[vec![0; 100], vec![0; 10]], 2, 100).is_err());
    }

    #[test]
    fn result_paths_separate_top_k_runs() {
        assert_eq!(
            result_path("ablation", "sift", 10),
            "visualizations/ablation_sift.json"
        );
        assert_eq!(
            result_path("ablation", "sift", 100),
            "visualizations/ablation_sift_k100.json"
        );
    }
}

/// 40 MB cache-flush trick — direct port of ParlayANN's
/// `check_nn_recall.h:46` pattern
/// (`auto volatile xx = parlay::random_permutation<long>(5000000);`).
///
/// Allocates a 5 M × 8 B = **40 MB** `Vec<u64>`, fills it sequentially
/// (write-touches every byte), then Fisher-Yates-shuffles it (random-
/// order touches every byte again). Apple M2 cache hierarchy is:
///
/// | level | size |
/// |---|---|
/// | L1 d-cache (per P-core) | 128 KB × 8 ≈ 1 MB total |
/// | L2 (P-cluster) | 16 MB shared |
/// | SLC (system-level) | ~24 MB |
///
/// Sum ≈ 41 MB, so the 40 MB shuffle evicts the entire working set —
/// every resident cache line from the previous timed region is gone by
/// the time we return. Subsequent graph / vector loads hit DRAM cold,
/// matching the "first query" scenario PA uses for its QPS numbers
/// instead of getting a "free" L2 hit from the prior iteration's
/// residuals.
///
/// Runtime: ~5 ms on an M2 Pro (one sequential init pass + one random-
/// order shuffle). Uses a fixed-seed xorshift RNG so the sequence is
/// deterministic — we only care that the access pattern is random from
/// the hardware prefetcher's point of view, not that the bytes differ
/// across calls.
///
/// `black_box` replaces the `volatile` keyword that PA uses — same
/// intent: forbid the optimizer from dead-code-eliminating the
/// allocation + touch-work.
#[inline(never)]
pub fn flush_cache() {
    const N: usize = 5_000_000;
    let mut v: Vec<u64> = (0..N as u64).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..N).rev() {
        // xorshift64 — sufficient for a cache-eviction access pattern.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
    black_box(&v);
}

/// Bump the calling thread's macOS QoS class to USER_INTERACTIVE.
/// Pinns the scheduler's bias toward P-cores and prevents demotion to
/// UTILITY under sustained CPU pressure or thermal events. Must be
/// called on each worker thread that needs the boost — set via
/// `rayon::ThreadPoolBuilder::start_handler` to apply pool-wide. Also
/// useful from the main thread driving `pool.install(...)`.
///
/// Best-effort: return code ignored (the call is advisory). No-op on
/// non-macOS targets.
#[cfg(target_os = "macos")]
#[inline]
pub fn set_thread_qos_user_interactive() {
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
pub fn set_thread_qos_user_interactive() {}

/// `mlock(2)` a byte range so its pages cannot be paged out. Used to
/// pin the dataset / quantized base / PhasedGraph slab / query batch
/// into RAM for timed sweeps so warmup + trials see exactly the same
/// resident pages — no page-in cost contaminating the early trials.
///
/// Best-effort: failure (e.g. `RLIMIT_MEMLOCK` exceeded, or non-Unix
/// target) is logged and the caller falls back to ordinary paging.
/// Same shape as the `mlock_bytes` helper in `orion.rs` — the
/// two can be unified once one of them stops moving.
#[cfg(unix)]
#[inline]
pub fn mlock_bytes(label: &str, ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    let rc = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        log::error!(
            "[mlock] {label}: failed to pin {} MiB ({err}) — falling back to pageable",
            len / (1024 * 1024)
        );
    } else {
        log::info!("[mlock] {label}: pinned {} MiB", len / (1024 * 1024));
    }
}

#[cfg(not(unix))]
#[inline]
pub fn mlock_bytes(_label: &str, _ptr: *const u8, _len: usize) {}
