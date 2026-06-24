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
        eprintln!(
            "[mlock] {label}: failed to pin {} MiB ({err}) — falling back to pageable",
            len / (1024 * 1024)
        );
    } else {
        println!("[mlock] {label}: pinned {} MiB", len / (1024 * 1024));
    }
}

#[cfg(not(unix))]
#[inline]
pub fn mlock_bytes(_label: &str, _ptr: *const u8, _len: usize) {}
