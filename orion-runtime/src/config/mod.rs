/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Configuration reading order: caller `options`, property-specific rules,
//! compatibility `plan`, then `resolved` execution input and cache identity.
//! External precedence belongs to driver adapters; these modules do not read
//! YAML, inspect the host, or choose cache files.

pub(crate) mod cascade;
mod data;
pub(crate) mod execution;
mod graph;
pub mod options;
pub(crate) mod plan;
mod resolved;
pub(crate) mod resources;

use data::SearchMetric;
pub use data::VectorStorageKind;
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub use execution::PreflightFormat;
pub use execution::ResolvedSweep;
pub use graph::{GraphSource, GraphSourcePolicy};
pub use resolved::ResolvedRunConfig;
