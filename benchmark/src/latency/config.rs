/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit experiment inputs; no dataset shortcuts in the public Index API.
use anyhow::{ensure, Result};
use orion_ann::{GraphSourcePolicy, IndexOptions, VectorFormat};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::PathBuf, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Experiment {
    pub dataset: String,
    pub base: PathBuf,
    pub query: PathBuf,
    pub groundtruth: Option<PathBuf>,
    pub query_format: Option<VectorFormat>,
    /// Use a verified public snapshot instead of preparing a working cache.
    pub snapshot: Option<PathBuf>,
    pub index: IndexOptions,
    pub k: usize,
    pub beams: Vec<usize>,
    #[serde(default)]
    pub scores: bool,
    pub calibration_samples: usize,
    pub calibration_l: usize,
    #[serde(default)]
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Protocol {
    pub concurrency: Vec<usize>,
    pub rounds: usize,
    pub min_requests: usize,
    pub min_seconds: f64,
    /// Checked between requests, never cancels an in-progress Index::search.
    pub max_seconds: f64,
    pub warmup_requests: usize,
    pub warmup_seconds: f64,
    pub cooldown_seconds: f64,
    pub seed: u64,
    /// Also measure one untimed control per point; never mix it into latency.
    pub overhead_check: bool,
}

impl Default for Protocol {
    fn default() -> Self {
        Self {
            concurrency: vec![1, 8],
            rounds: 3,
            min_requests: 10_000,
            min_seconds: 1.0,
            max_seconds: 30.0,
            warmup_requests: 1_000,
            warmup_seconds: 0.2,
            cooldown_seconds: 1.0,
            seed: 20261112,
            overhead_check: false,
        }
    }
}

impl Protocol {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.rounds > 0 && self.min_requests > 0,
            "rounds and min_requests must be positive"
        );
        ensure!(!self.concurrency.is_empty(), "concurrency list is empty");
        let unique: BTreeSet<_> = self.concurrency.iter().copied().collect();
        ensure!(
            unique.len() == self.concurrency.len(),
            "duplicate concurrency"
        );
        ensure!(
            unique.iter().all(|&c| (1..=64).contains(&c)),
            "concurrency must be in 1..=64"
        );
        for (name, value) in [
            ("min_seconds", self.min_seconds),
            ("max_seconds", self.max_seconds),
            ("warmup_seconds", self.warmup_seconds),
            ("cooldown_seconds", self.cooldown_seconds),
        ] {
            ensure!(
                value.is_finite() && (0.0..=3600.0).contains(&value),
                "{name} must be finite and in 0..=3600"
            );
        }
        ensure!(
            self.max_seconds > 0.0 && self.max_seconds >= self.min_seconds,
            "max_seconds must be positive and >= min_seconds"
        );
        Ok(())
    }

    pub(crate) fn minimum(&self) -> Duration {
        Duration::from_secs_f64(self.min_seconds)
    }
    pub(crate) fn maximum(&self) -> Duration {
        Duration::from_secs_f64(self.max_seconds)
    }
}

impl Experiment {
    pub fn validate(&self) -> Result<()> {
        self.protocol.validate()?;
        ensure!(
            self.snapshot.is_some()
                || (self.index.graph.source == GraphSourcePolicy::Parlayann
                    && self.index.graph.staged_file.is_some()),
            "use a snapshot or explicit parlayann source with a staged file; fresh graph construction is disabled"
        );
        ensure!(!self.dataset.trim().is_empty(), "dataset label is empty");
        ensure!(
            self.k > 0 && self.calibration_samples > 0,
            "k and calibration_samples must be positive"
        );
        ensure!(self.calibration_l >= self.k, "calibration L must be >= k");
        ensure!(
            !self.beams.is_empty() && self.beams.iter().all(|&l| l >= self.k),
            "beam list must be nonempty with L >= k"
        );
        ensure!(
            self.beams.iter().copied().collect::<BTreeSet<_>>().len() == self.beams.len(),
            "duplicate beam"
        );
        Ok(())
    }
}
