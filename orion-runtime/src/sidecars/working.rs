/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Working records map a base fingerprint and codec to verified objects.
//! Each live index tracks the objects materialized by its own cascade.
//! State is declared first, then service attachment and load/store operations;
//! input identity and lookup helpers follow those operations.
use super::{Receipts, SnapshotSidecars};
use crate::integrity::read_verified;
#[cfg(any(test, feature = "benchmark"))]
use crate::storage::object_dir;
use crate::storage::{put, PendingFile};
use crate::{Error, Result};
use orion_ann_core::{
    index::{DecodeSidecar, EncodeSidecar, IndexData, SidecarCache},
    model::dataset::rows::Rows,
    CompressedIndex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

/// One cache service shared by all codec slots of an index, not a single payload.
struct WorkingCache {
    directory: PathBuf,
    records: Receipts,
    metadata: Option<PathBuf>,
    input: CacheInput,
}

/// Reuse an authenticated input identity or fingerprint resident rows lazily.
enum CacheInput {
    Authenticated(String),
    Rows(Mutex<Option<String>>),
}

/// One codec's working lookup record binds input identity to a persisted object.
#[derive(Serialize, Deserialize)]
pub(super) struct Identity {
    pub(super) dataset_sha256: String,
    pub(super) payload_sha256: String,
}

/// Attach the shared cache service and return its session-wide receipts.
/// This does not read a sidecar: lazy slots call load/store with their codec.
/// Snapshot payloads have already been restored; `initial` retains their receipts.
/// A reopened index can persist additional codecs into its original object store.
pub(crate) fn track<D: IndexData>(
    index: &mut CompressedIndex<D>,
    directory: PathBuf,
    metadata: Option<PathBuf>,
    initial: SnapshotSidecars,
    derived: Option<String>,
) -> Receipts {
    let records = Arc::new(Mutex::new(initial));
    {
        index.set_sidecar_cache(Box::new(WorkingCache {
            directory,
            metadata,
            records: records.clone(),
            input: match derived {
                Some(key) => CacheInput::Authenticated(key),
                None => CacheInput::Rows(Mutex::default()),
            },
        }));
    }
    records
}

#[cfg(any(test, feature = "benchmark"))]
pub(crate) fn attach<D: IndexData>(index: &mut CompressedIndex<D>, metadata: &Path) -> Result<()> {
    let directory = object_dir(metadata.parent().unwrap_or(Path::new(".")))?;
    track(
        index,
        directory,
        Some(metadata.to_owned()),
        SnapshotSidecars::new(),
        None,
    );
    Ok(())
}

impl SidecarCache for WorkingCache {
    fn load(
        &self,
        base: &dyn Rows,
        codec: &'static str,
        decode: &mut DecodeSidecar<'_>,
    ) -> Result<()> {
        let Some(metadata) = &self.metadata else {
            return Ok(());
        };
        let path = metadata.with_extension(codec);
        if let Some(digest) = expected_digest(&path, &self.input.codec_key(base, codec)?) {
            let object = crate::storage::object(&self.directory, &digest, codec)?;
            let verified = read_verified(&object, codec, &digest, None, decode)?;
            self.records
                .lock()
                .map_err(|_| Error::invalid_state("sidecar receipt lock poisoned"))?
                .insert(codec.to_owned(), (verified.bytes, verified.digest));
        }
        Ok(())
    }

    fn store(
        &self,
        base: &dyn Rows,
        codec: &'static str,
        encode: &EncodeSidecar<'_>,
    ) -> Result<()> {
        let (bytes, digest) = put(&self.directory, codec, |writer| encode(writer))?;
        self.records
            .lock()
            .map_err(|_| Error::invalid_state("sidecar receipt lock poisoned"))?
            .insert(codec.to_owned(), (bytes, digest.clone()));
        let Some(metadata) = &self.metadata else {
            return Ok(());
        };
        let path = metadata.with_extension(codec);
        let identity = Identity {
            dataset_sha256: self.input.codec_key(base, codec)?,
            payload_sha256: digest,
        };
        let record = metadata_path(&path);
        let (temporary, mut writer) = PendingFile::create(&record)?;
        writer.write_all(&serde_json::to_vec(&identity)?)?;
        writer.sync_all()?;
        drop(writer);

        fs::rename(&temporary.0, record)?;
        Ok(())
    }
}

impl CacheInput {
    fn codec_key(&self, base: &dyn Rows, codec: &str) -> Result<String> {
        let slot = match self {
            Self::Authenticated(input) => {
                // Include the codec in the identity as well as the lookup filename.
                return Ok(crate::integrity::sha256(&serde_json::to_vec(&(
                    "orion/sidecar-codec-v1",
                    input,
                    codec,
                ))?));
            }
            Self::Rows(slot) => slot,
        };
        let mut cached = slot
            .lock()
            .map_err(|_| Error::invalid_state("sidecar fingerprint lock poisoned"))?;
        if let Some(digest) = cached.as_ref() {
            return Ok(digest.clone());
        }
        let digest = fingerprint(base)?;
        *cached = Some(digest.clone());
        Ok(digest)
    }
}

/// Identify the logical rows read by a codec from an already authenticated input.
pub(crate) fn derived_key(
    config: &crate::config::ResolvedRunConfig,
    points: usize,
) -> Result<Option<String>> {
    let Some(base) = &config.base_content_id else {
        return Ok(None);
    };
    let recipe = serde_json::to_vec(&(
        "orion/sidecar-input-v2",
        base,
        config.base_format,
        config.storage(),
        config.dimension,
        points,
        config.metric.identity_key(),
    ))?;
    Ok(Some(crate::integrity::sha256(&recipe)))
}

pub(super) fn fingerprint(dataset: &dyn Rows) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"orion-sidecar-base-v1");
    hash.update((dataset.dimension() as u64).to_le_bytes());
    hash.update((dataset.len() as u64).to_le_bytes());
    let mut scratch = Vec::new();
    for id in 0..dataset.len() {
        let row = dataset.row_f32(id, &mut scratch);
        if row.len() != dataset.dimension() {
            return Err(Error::dimension(
                "sidecar row",
                dataset.dimension(),
                row.len(),
            ));
        }
        // Hash logical coordinates and their order, never platform SIMD padding.
        #[cfg(target_endian = "little")]
        hash.update(bytemuck::cast_slice(row));
        #[cfg(target_endian = "big")]
        for value in row {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn metadata_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".identity.json");
    PathBuf::from(name)
}

fn expected_digest(path: &Path, dataset: &str) -> Option<String> {
    let mut bytes = Vec::new();
    File::open(metadata_path(path))
        .ok()?
        .take(4097)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 4096 {
        return None;
    }
    let identity: Identity = serde_json::from_slice(&bytes).ok()?;
    (identity.dataset_sha256 == dataset).then_some(identity.payload_sha256)
}
