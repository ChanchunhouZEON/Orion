/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::metrics::{memory::format_bytes, LatencyStats};
use comfy_table::{Cell, Table};
use std::time::Duration;

/// A single row of benchmark results.
pub struct BenchmarkResult {
    pub algorithm: String,
    pub params: String,
    pub build_time: Duration,
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub qps: f64,
    pub latency: LatencyStats,
    pub peak_memory: usize,
    /// Heap in use immediately after build (steady-state index footprint).
    pub index_memory: usize,
}

/// Print a formatted table of benchmark results.
pub fn print_results_table(results: &[BenchmarkResult]) {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Algorithm"),
        Cell::new("Params"),
        Cell::new("Build (s)"),
        Cell::new("R@1"),
        Cell::new("R@10"),
        Cell::new("R@100"),
        Cell::new("QPS"),
        Cell::new("Mean Lat (ms)"),
        Cell::new("P99 Lat (ms)"),
        Cell::new("Peak Mem"),
        Cell::new("Index Mem"),
    ]);

    for r in results {
        table.add_row(vec![
            Cell::new(&r.algorithm),
            Cell::new(&r.params),
            Cell::new(format!("{:.2}", r.build_time.as_secs_f64())),
            Cell::new(format!("{:.4}", r.recall_at_1)),
            Cell::new(format!("{:.4}", r.recall_at_10)),
            Cell::new(format!("{:.4}", r.recall_at_100)),
            Cell::new(format!("{:.1}", r.qps)),
            Cell::new(format!("{:.2}", r.latency.mean.as_secs_f64() * 1000.0)),
            Cell::new(format!("{:.2}", r.latency.p99.as_secs_f64() * 1000.0)),
            Cell::new(format_bytes(r.peak_memory)),
            Cell::new(format_bytes(r.index_memory)),
        ]);
    }

    println!("{table}");
}
