/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Execution records and bounds shared by service and driver adapters.
use crate::{Error, Result};

/// Presentation format for an execution memory forecast.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub enum PreflightFormat {
    /// Table for a terminal, JSON when stdout is redirected.
    #[default]
    Auto,
    Table,
    Json,
}

/// Validated execution schedule used by command-line and benchmark adapters.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedSweep {
    /// Requested neighbors per query for this execution schedule.
    pub k: usize,

    /// Benchmark schedule label; not part of production index behavior.
    #[cfg(any(feature = "benchmark", test))]
    pub profile: String,

    /// Beam capacities to evaluate after index preparation.
    pub search_list_sizes: Vec<usize>,

    /// Worker count selected for preparation and batch execution.
    pub threads: usize,

    /// Number of benchmark repetitions per search setting.
    #[cfg(any(feature = "benchmark", test))]
    pub trials: usize,

    /// Maximum query rows selected for explicit calibration.
    pub calibration_samples: usize,

    /// Beam capacity used to measure adaptive calibration parameters.
    pub calibration_l: usize,
}

/// Rejects a zero worker count before creating any thread pool.
pub fn validate_positive_threads(threads: usize) -> Result<()> {
    if threads == 0 {
        return Err(Error::configuration(
            "threads",
            "thread count must be positive",
        ));
    }
    Ok(())
}

/// Checks worker count against the core's supported maximum.
pub fn validate_threads(threads: usize) -> Result<()> {
    validate_positive_threads(threads)?;
    if threads > orion_ann_core::search::MAX_WORKERS {
        return Err(
            Error::configuration("threads", "thread count exceeds MAX_WORKERS")
                .with_detail("maximum", orion_ann_core::search::MAX_WORKERS)
                .with_detail("actual", threads),
        );
    }
    Ok(())
}

/// Checks the convergence observation-window range shared by frontends.
pub fn validate_window(window: usize) -> Result<()> {
    if !(1..=64).contains(&window) {
        return Err(Error::configuration(
            "window_size",
            "window_size must be in 1..=64",
        ));
    }
    Ok(())
}
