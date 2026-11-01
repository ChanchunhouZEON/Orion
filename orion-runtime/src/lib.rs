//! Shared Orion runtime: configuration, input, graph preparation and search.
//! CLI and Python use the same runtime without depending on each other.
//!
//! See the [API guide](guide) for configuration, build, cache and search examples,
//! including error, ownership and concurrency contracts.
//!
//! # Embedded Rust service
//!
//! [`Index`] owns its storage and Rayon pool; ordinary operations require no
//! process-wide pool or YAML configuration. [`orion_ann::Orion`] is the lower-level
//! API for callers supplying their own graph and storage. Both use [`Error`].
//!
//! ```no_run
//! use orion_ann_runtime::{BaseInput, Index, IndexOptions, Result};
//!
//! fn serve() -> Result<()> {
//!     let mut index = Index::new(
//!         BaseInput::File("base.fbin".into()),
//!         IndexOptions { threads: 8, cache_dir: "build-cache".into(), ..Default::default() },
//!     )?;
//!     let queries = vec![0.0_f32; index.dimension() * 20];
//!     index.calibrate(queries.clone(), 10, None)?;
//!     let results = index.search(queries, 10, 64, true)?;
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
pub mod config;
pub mod data;
pub mod graph;
pub mod guide;
pub mod index;
pub mod search;

pub use config::options::{CascadeConfig, GraphConfig, IndexOptions};
pub use index::{BaseInput, Calibration, Index, SearchResult};

pub use orion_ann::{Error, ErrorCode, ErrorReport, Result};

/// Portable diagnostics, also used by the WebAssembly replay interface.
pub use orion_ann_diagnostics as diagnostics;
