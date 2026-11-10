/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Portable publication. Object copies are verified before the directory is exposed.
use super::{manifest::Snapshot, OpenOptions, SnapshotRef, SnapshotSource};
use crate::{BaseInput, Error, Index, Result};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

impl Index {
    /// Copy this generation's immutable objects into a relocatable bundle.
    /// The returned reference names `destination/snapshot`; move it with
    /// [`SnapshotRef::relocated`]. `include_base` copies the original file, not
    /// the resident normalized/padded representation. Array-built indices must
    /// use `false` and supply the original array when reopening.
    ///
    /// # Errors
    /// Rejects existing destinations, corrupted objects, changed base files,
    /// insufficient disk space and `include_base` for array-built indices.
    /// Writes stream through a bounded buffer; the live index is unchanged.
    pub fn export_bundle(&self, destination: &Path, include_base: bool) -> Result<SnapshotRef> {
        if include_base && self.base_file.is_none() {
            return Err(Error::invalid_argument("including base requires a file-built index; export objects only and supply the original array on open"));
        }
        let destination = std::path::absolute(destination)?;
        let digest = crate::storage::publish(&destination, |stage| {
            let reference = self.save(&stage.join("snapshot"))?;
            let manifest_path = reference.path().join("index.json");
            let mut manifest: Snapshot = serde_json::from_slice(&fs::read(&manifest_path)?)?;
            let store = stage.join("objects");
            let files = &manifest.payloads;
            let mut objects = vec![
                ("pgraph", files.graph_sha256.as_str()),
                ("bin", files.metadata_sha256.as_str()),
                ("cache.json", files.provenance_sha256.as_str()),
            ];
            for (codec, (_, digest)) in &files.sidecars {
                objects.push((codec.as_str(), digest.as_str()));
            }
            if let Some(ids) = &files.external_ids {
                objects.push(("ids-u64-le", &ids.sha256));
            }
            for (codec, digest) in objects {
                let source = crate::storage::object(&self.view.object_store, digest, codec)?;
                let (_, copied) = crate::storage::put(&store, codec, |writer| {
                    std::io::copy(&mut fs::File::open(&source)?, writer)?;
                    Ok(())
                })?;
                crate::integrity::check("bundle object", &copied, digest)?;
            }
            if include_base {
                let encoded = crate::storage::encode_file(&stage.join("base"), |writer| {
                    std::io::copy(
                        &mut fs::File::open(self.base_file.as_ref().unwrap())?,
                        writer,
                    )?;
                    Ok(())
                })?;
                crate::integrity::check("bundle base", &encoded.digest, &files.base_sha256)?;
                fs::rename(&encoded.temporary.0, stage.join("base"))?;
            }
            manifest.bundled_base = include_base;
            manifest.object_store = PathBuf::from("../objects");
            let bytes = serde_json::to_vec_pretty(&manifest)?;
            let mut file = fs::File::create(&manifest_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            crate::storage::sync_dir(reference.path())?;
            Ok(crate::integrity::sha256(&bytes))
        })?;
        Ok(SnapshotRef {
            path: destination.join("snapshot"),
            manifest_sha256: digest,
        })
    }
}

impl OpenOptions {
    /// Open an exported bundle containing its base. `source` names the returned
    /// snapshot reference/directory, not the bundle root. All ordinary version,
    /// identity, shape and SHA-256 checks still apply.
    pub fn open_bundle(self, source: impl Into<SnapshotSource>) -> Result<Index> {
        let (path, manifest) = super::read_manifest(source.into())?;
        if !manifest.bundled_base {
            return Err(Error::invalid_argument(
                "bundle has no base; use open with the original base input",
            ));
        }
        let base = BaseInput::from_file(path.join("../base"));
        self.load_snapshot(&path, manifest, base)
    }
}
