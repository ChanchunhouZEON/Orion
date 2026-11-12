/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Aggregation is sample-weighted: merge distributions, never average percentiles.
use super::{
    evaluation::Evaluation,
    runner::Trial,
    statistics::{Bucket, Distribution, Percentiles},
};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct Point {
    pub beam: usize,
    pub concurrency: usize,
    pub evaluation: Evaluation,
    pub rounds: Vec<Trial>,
    pub pooled: Option<Percentiles>,
    pub histogram: Vec<Bucket>,
    pub p99_round_range_us: Option<[f64; 2]>,
    pub median_instrumented_qps: Option<f64>,
    pub untimed_control: Option<Trial>,
}
impl Point {
    pub fn new(beam: usize, concurrency: usize, evaluation: Evaluation) -> Self {
        Self {
            beam,
            concurrency,
            evaluation,
            rounds: vec![],
            pooled: None,
            histogram: vec![],
            p99_round_range_us: None,
            median_instrumented_qps: None,
            untimed_control: None,
        }
    }
    pub fn refresh(&mut self) -> Result<()> {
        let mut pooled = Distribution::new()?;
        let mut qps = vec![];
        let mut p99 = vec![];
        for trial in &self.rounds {
            ensure!(
                trial.latency.is_some(),
                "untimed controls cannot enter the latency distribution"
            );
            pooled.merge(&trial.distribution)?;
            qps.push(trial.instrumented_qps);
            p99.push(trial.latency.as_ref().unwrap().p99_us);
        }
        self.pooled = pooled.summary();
        self.histogram = pooled.buckets();
        self.median_instrumented_qps = median(&qps);
        self.p99_round_range_us = p99
            .iter()
            .copied()
            .reduce(f64::min)
            .zip(p99.iter().copied().reduce(f64::max))
            .map(|(lo, hi)| [lo, hi]);
        Ok(())
    }
}
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[m - 1] + values[m]) / 2.0
    } else {
        values[m]
    })
}

/// An interrupted run retains its last checkpoint with status=running.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.partial");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
