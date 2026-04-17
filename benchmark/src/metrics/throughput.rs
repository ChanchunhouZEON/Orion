/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::time::{Duration, Instant};

/// Measure queries per second for a batch of queries.
/// Returns (QPS, total_duration, individual_durations).
#[allow(dead_code)]
pub fn measure_qps<F>(num_queries: usize, mut search_fn: F) -> (f64, Duration, Vec<Duration>)
where
    F: FnMut(usize) -> Duration,
{
    let mut durations = Vec::with_capacity(num_queries);
    let start = Instant::now();

    for i in 0..num_queries {
        let d = search_fn(i);
        durations.push(d);
    }

    let total = start.elapsed();
    let qps = num_queries as f64 / total.as_secs_f64();

    (qps, total, durations)
}
