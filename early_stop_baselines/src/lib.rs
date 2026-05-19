/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Early-stop ANN search baselines implemented against
//! [`diskann::index::InmemIndex`]. Each baseline is a standalone beam-search
//! variant over the DiskANN graph with a different termination rule —
//! intended for paper-table comparisons against our τ/ee method.
//!
//! Baselines provided:
//! - [`patience`]: "stop after K consecutive steps with no new admission"
//!
//! Future: `laet`, `adaptnn` (GBDT models via ONNX runtime).
//!
//! All search functions are free functions that accept:
//! - `&InmemIndex<T, N>` — read-only access to graph, dataset, entry point
//! - `&mut BaselineScratch` — caller-managed per-thread working buffers
//!
//! The crate does not depend on `staged_diskann`; it references only the
//! DiskANN public API surface.

pub mod patience;
pub mod scratch;

pub use patience::{PatienceChecker, search_patience, search_patience_batch};
pub use scratch::BaselineScratch;
