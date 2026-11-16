/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Configuration reading order: caller `options`, property-specific rules,
//! compatibility `plan`, then `resolved` execution input and cache identity.
//! External precedence belongs to driver adapters; these modules do not read
//! files, inspect the host, or choose cache files. Explicit YAML decoding uses
//! the same typed options as Rust callers.

pub(crate) mod cascade;
mod data;
pub(crate) mod execution;
mod graph;
pub mod options;
mod phase;
pub(crate) mod plan;
mod resolved;
pub(crate) mod resources;
mod yaml;

use data::SearchMetric;
pub use data::VectorStorageKind;
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub use execution::PreflightFormat;
pub use execution::ResolvedSweep;
pub use graph::{GraphSource, GraphSourcePolicy};
pub(crate) use phase::RunPhase;
pub use resolved::ResolvedRunConfig;
