/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::time::Duration;

/// Latency percentile statistics.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
    #[allow(dead_code)]
    pub min: Duration,
    #[allow(dead_code)]
    pub max: Duration,
}

impl LatencyStats {
    /// Compute latency statistics from a list of individual query durations.
    pub fn from_durations(mut durations: Vec<Duration>) -> Self {
        assert!(!durations.is_empty(), "Need at least one measurement");
        durations.sort();

        let n = durations.len();
        let total: Duration = durations.iter().sum();
        let mean = total / n as u32;

        Self {
            mean,
            p50: durations[n * 50 / 100],
            p95: durations[n * 95 / 100],
            p99: durations[n * 99 / 100],
            p999: durations[(n as f64 * 0.999).min((n - 1) as f64) as usize],
            min: durations[0],
            max: durations[n - 1],
        }
    }
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean={:.2}ms p50={:.2}ms p95={:.2}ms p99={:.2}ms p99.9={:.2}ms",
            self.mean.as_secs_f64() * 1000.0,
            self.p50.as_secs_f64() * 1000.0,
            self.p95.as_secs_f64() * 1000.0,
            self.p99.as_secs_f64() * 1000.0,
            self.p999.as_secs_f64() * 1000.0,
        )
    }
}
