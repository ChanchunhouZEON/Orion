/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! A session retains receipts for persisted sidecars used by its resident graph.
pub(crate) mod requirements;
#[cfg(feature = "benchmark")]
pub(crate) mod trusted;
mod working;
use crate::{Error, Result, integrity::read_verified};
use orion_ann_core::{CompressedIndex, index::IndexData};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
#[cfg(any(test, feature = "benchmark"))]
pub(crate) use working::attach;
pub(crate) use working::{derived_key, track};
pub(crate) type SnapshotSidecars = std::collections::BTreeMap<String, (u64, String)>;
pub(crate) type Receipts = Arc<Mutex<SnapshotSidecars>>;

/// The core can still serve searches if persisting an optional sidecar failed.
/// Refuse to snapshot such a session rather than silently omit the payload.
pub(crate) fn materialized<D: IndexData>(
    index: &CompressedIndex<D>,
    records: &Receipts,
) -> Result<SnapshotSidecars> {
    let receipts = records
        .lock()
        .map_err(|_| Error::invalid_state("sidecar receipt lock poisoned"))?;
    let mut used = SnapshotSidecars::new();
    index.visit_sidecars(|codec, _| {
        let record = receipts.get(codec).ok_or_else(|| {
            Error::cache(format!("materialized sidecar {codec} was not persisted"))
        })?;
        used.insert(codec.to_owned(), record.clone());
        Ok(())
    })?;
    Ok(used)
}

pub(crate) fn restore_snapshot<D: IndexData>(
    index: &mut CompressedIndex<D>,
    directory: &Path,
    records: &SnapshotSidecars,
) -> Result<()> {
    for (codec, (bytes, digest)) in records {
        // Core resolves the codec against its closed registry before this name
        // is used as a file extension. Verify on the handle used to decode it.
        index.restore_sidecar(codec, |decode| {
            let path = crate::storage::object(directory, digest, codec)?;
            read_verified(&path, codec, digest, Some(*bytes), decode).map(|v| v.value)
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
