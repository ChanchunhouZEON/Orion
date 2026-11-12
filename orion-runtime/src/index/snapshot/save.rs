/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Publish the manifest for one resident generation.
use super::{
    manifest::{check_lengths, execution_signature, Snapshot},
    SnapshotRef,
};
use crate::{
    config::{GraphSource, GraphSourcePolicy},
    Error, Index, Result,
};
use std::{fs, path::Path};
impl Index {
    /// See the [complete example and contract](crate::guide::cache).
    ///
    /// Publish one manifest into a new directory. Reopening requires the shared
    /// object store and the external base.
    ///
    /// # Errors
    /// Returns an error for an existing destination, changed base identity,
    /// serialization or I/O failure. The original loaded index remains usable.
    pub fn save(&self, destination: &Path) -> Result<SnapshotRef> {
        let path = std::path::absolute(destination)?;
        if destination.try_exists().map_err(Error::from)? {
            return Err(Error::invalid_state(
                "save destination already exists; choose a new directory",
            )
            .at("save")
            .with_detail("path", destination.to_string_lossy().into_owned()));
        }
        if let Some(path) = &self.base_file {
            if crate::integrity::file_sha256(path)? != self.view.payloads.base_sha256 {
                return Err(
                    Error::cache("external base content changed since index was loaded")
                        .with_detail("role", "base"),
                );
            }
        }
        let label_snapshot = self.labels.snapshot()?;
        crate::storage::publish(&path, |staging| {
            let config = &self.config;
            let mut files = self.view.payloads.clone();
            files.sidecars = self.view.engine.sidecars()?;
            files.labels = if label_snapshot.revision == 0 {
                None
            } else {
                Some(label_snapshot.save(&self.view.object_store)?)
            };
            check_lengths(
                &self.view.object_store,
                &files,
                self.view.graph_bytes,
                self.view.metadata_bytes,
            )?;
            for (codec, (len, sha)) in &files.sidecars {
                if fs::metadata(crate::storage::object(&self.view.object_store, sha, codec)?)?.len()
                    != *len
                {
                    return Err(Error::cache("snapshot sidecar length mismatch"));
                }
            }
            let mut options = self.options.clone();
            // Persist the resolved graph source. Reopening is never automatic.
            options.graph.source = match config.graph_source {
                GraphSource::Rust => GraphSourcePolicy::Rust,
                GraphSource::Parlayann => GraphSourcePolicy::Parlayann,
            };
            options.graph.staged_file = None;
            options.storage = Some(self.storage());
            options.base_format = Some(config.base_format);
            let snapshot = Snapshot {
                format_version: super::manifest::FORMAT_VERSION,
                object_store: self.view.object_store.clone(),
                bundled_base: false,
                graph_bytes: self.view.graph_bytes,
                metadata_bytes: self.view.metadata_bytes,
                payloads: files,
                options,
                calibrations: self.calibrations.clone(),
                namespace: config.cache_namespace.clone(),
                execution_signature: execution_signature(config)?,
            };
            let file = fs::File::create(staging.join("index.json")).map_err(Error::from)?;
            let bytes = serde_json::to_vec_pretty(&snapshot)?;
            std::io::Write::write_all(&mut &file, &bytes)?;
            file.sync_all()?;
            Ok(crate::integrity::sha256(&bytes))
        })
        .map(|manifest_sha256| SnapshotRef {
            path,
            manifest_sha256,
        })
    }
}
