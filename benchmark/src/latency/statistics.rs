/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Fixed-range per-client histograms; all merging and JSON conversion is offline.
use anyhow::{ensure, Context, Result};
use hdrhistogram::Histogram;
use serde::Serialize;

const MAX_NS: u64 = 3_600_000_000_000;

#[derive(Debug, Clone)]
pub struct Distribution {
    histogram: Histogram<u64>,
    sum: u128,
    maximum: u64,
    zeros: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Percentiles {
    pub samples: u64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub zero_clock_samples: u64,
}

#[derive(Debug, Serialize)]
pub struct Bucket {
    pub lower_ns: u64,
    pub upper_ns: u64,
    pub count: u64,
}

impl Distribution {
    pub fn new() -> Result<Self> {
        Ok(Self {
            histogram: Histogram::new_with_bounds(1, MAX_NS, 3)?,
            sum: 0,
            maximum: 0,
            zeros: 0,
        })
    }

    pub fn record(&mut self, nanoseconds: u64) -> Result<()> {
        // Never resize or clip an outlier silently. Zero indicates clock
        // resolution; HDR supports recording it without inventing a 1 ns sample.
        ensure!(
            nanoseconds <= MAX_NS,
            "request exceeds the one-hour histogram range"
        );
        self.histogram
            .record(nanoseconds)
            .context("record request latency")?;
        self.sum += nanoseconds as u128;
        self.maximum = self.maximum.max(nanoseconds);
        self.zeros += u64::from(nanoseconds == 0);
        Ok(())
    }

    pub fn merge(&mut self, other: &Self) -> Result<()> {
        self.histogram.add(&other.histogram)?;
        self.sum += other.sum;
        self.maximum = self.maximum.max(other.maximum);
        self.zeros += other.zeros;
        Ok(())
    }

    pub fn summary(&self) -> Option<Percentiles> {
        let samples = self.histogram.len();
        (samples > 0).then(|| Percentiles {
            samples,
            mean_us: self.sum as f64 / samples as f64 / 1000.0,
            p50_us: self.histogram.value_at_quantile(0.50) as f64 / 1000.0,
            p95_us: self.histogram.value_at_quantile(0.95) as f64 / 1000.0,
            p99_us: self.histogram.value_at_quantile(0.99) as f64 / 1000.0,
            max_us: self.maximum as f64 / 1000.0,
            zero_clock_samples: self.zeros,
        })
    }

    pub fn buckets(&self) -> Vec<Bucket> {
        self.histogram
            .iter_recorded()
            .map(|entry| {
                let value = entry.value_iterated_to();
                Bucket {
                    lower_ns: self.histogram.lowest_equivalent(value),
                    upper_ns: self.histogram.highest_equivalent(value),
                    count: entry.count_at_value(),
                }
            })
            .collect()
    }
}
