/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve a snapshot directory or trusted reference and open its generation.
mod manifest;
mod save;
use crate::{BaseInput, Error, Index, Result};
pub(super) use manifest::{CheckedManifest, Payloads, Snapshot};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
/// A snapshot's location and the manifest identity recorded when it was saved.
/// Serialize this reference in trusted application state to reopen after a restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotRef {
    path: PathBuf,
    manifest_sha256: String,
}

impl SnapshotRef {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

/// Open the snapshot currently in a directory, or require the identity issued by save.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SnapshotSource {
    Directory(PathBuf),
    Reference(SnapshotRef),
}

impl<P: Into<PathBuf>> From<P> for SnapshotSource {
    fn from(path: P) -> Self {
        Self::Directory(path.into())
    }
}

impl From<SnapshotRef> for SnapshotSource {
    fn from(reference: SnapshotRef) -> Self {
        Self::Reference(reference)
    }
}

impl From<&SnapshotRef> for SnapshotSource {
    fn from(reference: &SnapshotRef) -> Self {
        Self::Reference(reference.clone())
    }
}

/// Resource overrides for reopening a snapshot. Defaults use the saved settings.
#[derive(Default, Clone, Copy, Debug)]
pub struct OpenOptions {
    pub threads: Option<usize>,
    pub memory_budget_gib: Option<f64>,
    /// Override the saved deployment policy; resources are always probed afresh.
    pub auto_pin: Option<bool>,
}

impl OpenOptions {
    /// Open through the same integrity checks as [`Index::open`].
    pub fn open(self, source: impl Into<SnapshotSource>, input: BaseInput) -> Result<Index> {
        let source = source.into();
        let path = match &source {
            SnapshotSource::Directory(path) => path.as_path(),
            SnapshotSource::Reference(reference) => reference.path(),
        };
        let bytes = crate::storage::read_bounded(&path.join("index.json"), 16 * 1024 * 1024)?;
        if let SnapshotSource::Reference(reference) = &source {
            crate::integrity::check(
                "manifest",
                &crate::integrity::sha256(&bytes),
                reference.manifest_sha256(),
            )?;
        }
        let mut snapshot: Snapshot = serde_json::from_slice(&bytes)?;
        // Standalone snapshots from the earlier layout already contain objects
        // under their content names. Their directory can serve as their store.
        if snapshot.object_store.as_os_str().is_empty() {
            snapshot.object_store = path.canonicalize()?;
        }
        if !snapshot.object_store.is_absolute() {
            return Err(Error::cache(
                "snapshot object store must be an absolute path",
            ));
        }
        let mut options = snapshot.options.clone();
        options.cache_dir = path.canonicalize().map_err(Error::from)?;
        if let Some(n) = self.threads {
            options.threads = n;
        }
        if self.memory_budget_gib.is_some() {
            options.memory_budget_gib = self.memory_budget_gib;
        }
        if let Some(auto_pin) = self.auto_pin {
            options.auto_pin = auto_pin;
        }
        Index::load(input, options, Some(snapshot))
    }
}
