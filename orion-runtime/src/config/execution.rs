/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Execution records and bounds shared by service and driver adapters.
use crate::{Error, Result};

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

#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedSweep {
    pub k: usize,
    #[cfg(any(feature = "benchmark", test))]
    pub profile: String,
    pub search_list_sizes: Vec<usize>,
    pub threads: usize,
    #[cfg(any(feature = "benchmark", test))]
    pub trials: usize,
    pub calibration_samples: usize,
    pub calibration_l: usize,
}

pub fn validate_positive_threads(threads: usize) -> Result<()> {
    if threads == 0 {
        return Err(Error::configuration(
            "threads",
            "thread count must be positive",
        ));
    }
    Ok(())
}

pub fn validate_threads(threads: usize) -> Result<()> {
    validate_positive_threads(threads)?;
    if threads > orion_ann_core::algorithm::search::MAX_WORKERS {
        return Err(
            Error::configuration("threads", "thread count exceeds MAX_WORKERS")
                .with_detail("maximum", orion_ann_core::algorithm::search::MAX_WORKERS)
                .with_detail("actual", threads),
        );
    }
    Ok(())
}

pub fn validate_window(window: usize) -> Result<()> {
    if !(1..=64).contains(&window) {
        return Err(Error::configuration(
            "window_size",
            "window_size must be in 1..=64",
        ));
    }
    Ok(())
}
