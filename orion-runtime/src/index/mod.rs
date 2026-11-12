/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! An owned index session for embedded applications. No CLI parsing or process-wide pool.
//!
//! `Index` below is the ownership boundary. Read `prepare::Index::load` next:
//! it turns `base::BaseInput` into a resident `view::ReadView`. Calibration and
//! query operations borrow that view; snapshot save records the same generation.
//! Each operation module declares its input/result types before its entry points,
//! with decoding, validation, and dispatch helpers below the main flow.
mod base;
mod calibration;
mod generation;
mod ids;
mod labels;
pub use generation::IndexSlot;
mod prepare;
mod query;
mod residency;
mod result;
mod snapshot;
mod view;
use crate::{
    config::{ResolvedRunConfig, VectorStorageKind},
    IndexOptions, Result,
};
pub use base::BaseInput;
pub use calibration::Calibration;
pub use query::{QueryBatch, SearchOptions};
pub use result::SearchResult;
pub use snapshot::{OpenOptions, SnapshotRef, SnapshotSource};
use std::{collections::BTreeMap, path::PathBuf};
use view::ReadView;

/// Each session owns a dedicated Rayon pool shared by concurrent searches.
/// Search borrows `&self`; calibration requires `&mut self`. Graph and vector
/// storage remain read-only while query-local scratch is checked out exclusively.
pub struct Index {
    view: ReadView,
    pool: rayon::ThreadPool,
    config: ResolvedRunConfig,
    options: IndexOptions,
    base_file: Option<PathBuf>,
    calibrations: BTreeMap<usize, Calibration>,
    external_ids: Option<ids::IdMap>,
    working_directory: Option<PathBuf>,
    labels: crate::labels::LabelStore,
}

impl Index {
    /// See the [complete example and contract](crate::guide::build).
    ///
    /// Prepare an owned index from a file or copied row-major array.
    ///
    /// # Errors
    /// Returns structured input/configuration, resource, I/O or backend errors.
    /// Invalid inputs are rejected before graph construction where possible.
    pub fn new(input: BaseInput, options: IndexOptions) -> Result<Self> {
        Self::load(input, options, None)
    }

    /// Construct with one unique application u64 ID per indexed row. The graph
    /// remains u32 and the mapping is saved independently of graph cache identity.
    /// The owned mapping is retained; validation briefly allocates a sorting copy.
    ///
    /// # Errors
    /// Rejects duplicate IDs, a count different from the indexed prefix, invalid
    /// configuration, insufficient estimated memory, writer conflicts and the
    /// same input/build/I/O failures as [`Index::new`].
    pub fn new_with_ids(input: BaseInput, options: IndexOptions, ids: Vec<u64>) -> Result<Self> {
        let count = ids.len();
        let mapping = ids::IdMap::validate(ids, count, options.memory_budget_gib)?;
        Self::load_with_ids(input, options, None, Some(mapping))
    }

    /// See the [complete example and contract](crate::guide::cache).
    ///
    /// Load and verify every snapshot payload. A directory selects the manifest
    /// at that path; a [`SnapshotRef`] also pins the manifest itself.
    /// Use [`OpenOptions::open`] to override thread count or memory budget.
    ///
    /// # Errors
    /// Returns an error on missing files, checksum or shape mismatches, invalid
    /// configuration, or insufficient memory. A miss never builds a graph.
    pub fn open(source: impl Into<SnapshotSource>, input: BaseInput) -> Result<Self> {
        OpenOptions::default().open(source, input)
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    pub fn storage(&self) -> VectorStorageKind {
        self.config.storage()
    }
    pub fn len(&self) -> usize {
        self.config.num_points.unwrap().min(self.config.max_points)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"num_points": self.len(), "dimension": self.dimension(),
        "storage": self.storage(), "metric": self.config.metric,
        "graph_source": self.config.graph_source, "graph_selection_reason": self.config.graph_selection_reason,
        "cascade": self.config.cascade(), "threads": self.options.threads,
        "cache_namespace": self.config.cache_namespace, "calibrations": self.calibrations,
        "mapped_sidecar_bytes": self.view.engine.mapped_sidecar_bytes(),
        "mapped_base_bytes": self.view.engine.mapped_base_bytes(),
        "mapped_graph_bytes": self.view.engine.mapped_graph_bytes(),
        "storage_options": self.options.storage_options,
        "external_ids_bytes": self.external_ids.as_ref().map_or(0, |ids| ids.0.len() * 8),
        "residency": self.view.engine.pin_report()})
    }
}

impl std::fmt::Debug for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Index")
            .field("points", &self.len())
            .field("dimension", &self.dimension())
            .field("threads", &self.pool.current_num_threads())
            .field("graph_source", &self.config.graph_source)
            .field("calibration_count", &self.calibrations.len())
            .finish_non_exhaustive()
    }
}
