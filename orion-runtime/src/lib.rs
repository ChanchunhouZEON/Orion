/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Shared Orion runtime: configuration, input, graph preparation and search.
//! CLI and Python use the same runtime without depending on each other.
//!
//! See the [API guide](guide) for configuration, build, cache and search examples,
//! including error, ownership and concurrency contracts.
//!
//! # Source reading order
//!
//! Start with `config/options.rs` for caller inputs, then `index/mod.rs` for
//! session ownership and `index/prepare.rs` for the load/build orchestration.
//! Follow `index/base.rs` for input validation and `graph/mod.rs` for graph
//! preparation. Once resident, `index/calibration.rs` and `index/query.rs`
//! dispatch through `index/view.rs`; `index/snapshot.rs` opens saved sessions.
//! The `sidecars`, `storage`, and `integrity` modules supply persistence services
//! to these operations. Types stay with the module that owns their invariants.
//!
//! # Embedded Rust service
//!
//! [`Index`] owns its storage and Rayon pool; ordinary operations require no
//! process-wide pool or YAML configuration. [`orion_ann_core::Orion`] is the lower-level
//! API for callers supplying their own graph and storage. Both use [`Error`].
//!
//! ```no_run
//! use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions, Result};
//!
//! fn serve() -> Result<()> {
//!     let mut index = Index::new(
//!         BaseInput::from_file("base.fbin"),
//!         IndexOptions { threads: 8, cache_dir: "build-cache".into(), ..Default::default() },
//!     )?;
//!     let queries = vec![0.0_f32; index.dimension() * 20];
//!     index.calibrate(queries.clone(), 10, None)?;
//!     let results = index.search(QueryBatch::from_f32(&queries, index.dimension())?, &SearchOptions::new(10, 64))?;
//!     assert_eq!(results.counts.len(), 20);
//!     index.save(std::path::Path::new("snapshot"))?;
//!     Ok(())
//! }
//! ```
//!
//! # Errors
//!
//! Public fallible operations return [`Result`]. Inspect [`Error::code`] and
//! [`Error::details`] for control flow, [`Error::report`] for JSON/FFI transport,
//! and `std::error::Error::source` for the original backend/I/O cause.
//! Constructing or propagating errors does not log. Applications choose their
//! own reporting policy; the CLI reports once at its outer boundary.
//!
//! User-input and file failures return errors. Internal invariant failures are
//! still programming bugs: Rust hosts choose their own panic policy. Python's
//! boundary catches unwinding panics, invalidates poisoned handles, and never
//! promises recovery from aborts or process-level allocation failures.
#![cfg_attr(
    not(feature = "benchmark"),
    doc = "
Experimental cache controls require the `benchmark` feature.
```compile_fail
use orion_ann::benchmark::utils::flush_cache;
```
"
)]
//!
//! Storage and graph-loading internals are private even when building adapters.
//! ```compile_fail,E0603
//! use orion_ann::data::LoadedBase;
//! ```
//! ```compile_fail,E0603
//! use orion_ann::config::ResolvedRunConfig;
//! ```
#![warn(missing_debug_implementations)]

mod config;
mod data;
mod graph;
mod integrity;
mod preflight;
mod residency;
mod sidecars;
pub use orion_ann_core::model::dataset::quantized_dataset::QuantizedKind;
pub use residency::{DatasetPin, PinHint, PinMode, PinOptions, PinReport, PinnedDatasets};
mod storage;

pub mod guide;
mod index;
mod search;
pub use config::cascade::SearchMetric;
/// Interfaces used by the independently built experiment executables.
#[cfg(feature = "benchmark")]
pub mod benchmark;
#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod execution;
pub use config::options::{CascadeConfig, GraphConfig, IndexOptions};
pub use config::{GraphSourcePolicy, VectorStorageKind};
#[cfg(feature = "file-queries")]
pub use data::query::read_query_file;
pub use data::VectorFormat;
pub use index::{
    BaseInput, Calibration, Index, OpenOptions, QueryBatch, SearchOptions, SearchResult,
    SnapshotRef, SnapshotSource,
};

pub use orion_ann_core::{Error, ErrorCode, ErrorReport, Result};

/// Portable diagnostics, also used by the WebAssembly replay interface.
pub mod diagnostics {
    pub use orion_ann_diagnostics::{
        Candidate, Phase, Score, SearchTrace, SpecialScore, StopReason, TraceEvent, TraceLimits,
        TraceMetadata, TraceReplay,
    };
}

/// Deployment policy for independently pageable product-quantization codes.
pub use orion_ann_core::model::dataset::quantized_dataset::pq::PqCodeStorage;
