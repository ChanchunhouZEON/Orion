/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Experiment-only cache decoding. No authenticated receipts leave this module.
//! The caller selects the base/recipe; core still validates shapes and edges.
use super::cache::CachePlan;
use crate::{Error, Result};
use orion_ann_core::{index::compressed_index::IndexData, CompressedIndex};
use orion_ann_platform::buffered_io::FILE_WRITE_BYTES;
use std::{
    fs::File,
    io::{BufReader, Cursor},
    path::Path,
};

pub(super) fn load<D: IndexData>(
    dataset: D,
    cache: CachePlan,
    config: &crate::config::ResolvedRunConfig,
) -> Result<CompressedIndex<D>> {
    let graph = cache.graph_path()?;
    let metadata = match &cache.selected {
        Some(generation) => crate::storage::object(
            &cache.objects()?,
            &generation.manifest.metadata_sha256,
            "bin",
        )?,
        None => cache.metadata_path.clone(),
    };
    log::info!(
        "Benchmark trusted cache: {} (SHA-256 skipped; layout validated)",
        graph.display()
    );
    let storage = crate::placement::graph_options(
        config,
        &dataset,
        std::fs::metadata(&graph)?.len().saturating_sub(16),
    )?;
    let mut index = decode_with_storage(&graph, &metadata, dataset, cache.legacy, &storage)?;
    crate::sidecars::trusted::attach(&mut index, &cache.metadata_path)?;
    Ok(index)
}

pub(super) fn decode<D: IndexData>(
    graph: &Path,
    metadata: &Path,
    dataset: D,
    legacy: bool,
) -> Result<CompressedIndex<D>> {
    decode_with_storage(graph, metadata, dataset, legacy, &Default::default())
}

fn decode_with_storage<D: IndexData>(
    graph: &Path,
    metadata: &Path,
    dataset: D,
    legacy: bool,
    storage: &orion_ann_platform::region::RegionOptions,
) -> Result<CompressedIndex<D>> {
    let mut header = crate::storage::read_bounded_file(File::open(metadata)?, metadata, 4096)?;
    // V5 is precisely the version/entry prefix; its slab format is unchanged.
    // Supply the missing dataset shape at this benchmark boundary, not in core's
    // verified decoder. Unknown formats remain errors, never guessed.
    if legacy && header.len() == 8 && header[..4] == 5u32.to_le_bytes() {
        header[..4].copy_from_slice(&6u32.to_le_bytes());
        header.extend_from_slice(&(dataset.dimension() as u64).to_le_bytes());
        header.extend_from_slice(&(dataset.point_count() as u64).to_le_bytes());
        let element = dataset.element_type().as_bytes();
        header.extend_from_slice(&(element.len() as u64).to_le_bytes());
        header.extend_from_slice(element);
    }
    let mut metadata_reader = Cursor::new(&header);
    let file = File::open(graph)?;
    let bytes = file.metadata()?.len();
    let index = CompressedIndex::decode_native_graph_with_storage(
        dataset,
        &mut metadata_reader,
        BufReader::with_capacity(FILE_WRITE_BYTES, file),
        bytes,
        storage,
    )?;
    if metadata_reader.position() != header.len() as u64 {
        return Err(Error::cache("trailing graph metadata bytes"));
    }
    Ok(index)
}
