/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::str::FromStr;

/// Distance metric
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Metric {
    /// Squared Euclidean (L2-Squared)
    L2,

    /// Cosine similarity
    Cosine,
}

#[derive(thiserror::Error, Debug)]
pub enum ParseMetricError {
    #[error("Invalid format for Metric: {0}")]
    InvalidFormat(String),
}

impl FromStr for Metric {
    type Err = ParseMetricError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" => Ok(Metric::L2),
            "cosine" => Ok(Metric::Cosine),
            _ => Err(ParseMetricError::InvalidFormat(String::from(s))),
        }
    }
}
