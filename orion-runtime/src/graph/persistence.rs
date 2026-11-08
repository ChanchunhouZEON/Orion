/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Filesystem boundary for native graphs. Core validates the binary layout.
use crate::integrity::read_verified;
#[cfg(any(test, feature = "benchmark"))]
use crate::storage::{encode_file, PendingFile};
#[cfg(any(test, feature = "benchmark"))]
use crate::Error;
use crate::Result;
use orion_ann_core::{index::compressed_index::IndexData, CompressedIndex};
use std::path::Path;
#[cfg(any(test, feature = "benchmark"))]
use std::{fs, io::Write};
#[cfg(any(test, feature = "benchmark"))]
pub(crate) fn save_graph<D: IndexData, S: Default>(
    index: &CompressedIndex<D, S>,
    path: &Path,
) -> Result<String> {
    if path.file_name().is_none() {
        return Err(
            Error::invalid_argument("cache path must name a metadata file").at("save_cache"),
        );
    }
    let graph_path = path.with_extension("pgraph");
    if graph_path == path {
        return Err(
            Error::invalid_argument("graph metadata and payload require distinct paths")
                .at("save_cache"),
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let (meta_tmp, mut metadata) = PendingFile::create(path)?;
    let mut header = Vec::new();
    let graph = encode_file(&graph_path, |writer| {
        index.encode_native_graph(&mut header, writer)
    })?;
    metadata.write_all(&header)?;
    metadata.sync_all()?;
    drop(metadata);
    // Readers check both payloads against the manifest, which is published last.
    fs::rename(&graph.temporary.0, graph_path)?;
    fs::rename(&meta_tmp.0, path)?;
    Ok(graph.digest)
}

#[cfg(test)]
pub(crate) fn load_verified<D: IndexData, S: Default>(
    path: &Path,
    dataset: D,
    metadata: &[u8],
    expected: &str,
) -> Result<crate::integrity::Verified<CompressedIndex<D, S>>> {
    load_verified_with_storage(path, dataset, metadata, expected, &Default::default())
}

pub(crate) fn load_verified_with_storage<D: IndexData, S: Default>(
    path: &Path,
    dataset: D,
    metadata: &[u8],
    expected: &str,
    storage: &orion_ann_platform::region::RegionOptions,
) -> Result<crate::integrity::Verified<CompressedIndex<D, S>>> {
    read_verified(path, "graph", expected, None, |reader, bytes| {
        CompressedIndex::decode_native_graph_with_storage(dataset, metadata, reader, bytes, storage)
    })
}

/// Publish both graph files before associating the sidecars with this cache.
#[cfg(any(test, feature = "benchmark"))]
pub fn save_working<D: IndexData>(index: &mut CompressedIndex<D>, path: &Path) -> Result<String> {
    let digest = save_graph(index, path)?;
    crate::sidecars::attach(index, path)?;
    Ok(digest)
}

/// Legacy standalone benchmark caches have no runtime manifest.
#[cfg(feature = "benchmark")]
pub fn load_working<D: IndexData>(path: &Path, dataset: D) -> Result<CompressedIndex<D>> {
    let mut index = super::trusted::decode(&path.with_extension("pgraph"), path, dataset, true)?;
    crate::sidecars::trusted::attach(&mut index, path)?;
    Ok(index)
}

#[cfg(test)]
mod tests;
