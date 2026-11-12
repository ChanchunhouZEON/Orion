/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Restore the verified immutable graph and sidecars without builder fallback.
//! Placement is resolved before allocations; receipts bind the restored objects
//! to this generation so later saves can reuse their digests.
use super::super::{
    snapshot::CheckedManifest,
    view::{DynamicEngine, ReadView},
};
use crate::{
    config::ResolvedRunConfig,
    data::{BaseElement, LoadedBase},
    Result,
};

pub(super) fn restore<T: BaseElement + 'static>(
    config: &ResolvedRunConfig,
    mut data: LoadedBase<T>,
    saved: CheckedManifest,
) -> Result<ReadView>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    let files = saved.payloads;
    let path = crate::storage::object(&saved.store, &files.graph_sha256, "pgraph")?;
    let dataset = data.take_index_dataset()?;
    let storage = crate::placement::graph_options(
        config,
        &dataset,
        std::fs::metadata(&path)?.len().saturating_sub(16),
    )?;
    let graph = crate::graph::persistence::load_verified_with_storage(
        &path,
        dataset,
        &saved.metadata,
        &files.graph_sha256,
        &storage,
    )?;
    let mut index = graph.value;
    crate::search::bind_rows(&index, config)?;
    crate::placement::configure(&index, config, files.sidecars.keys().cloned())?;
    crate::sidecars::restore_snapshot(&mut index, &saved.store, &files.sidecars)?;
    let receipts = crate::sidecars::track(
        &mut index,
        saved.store.clone(),
        None,
        files.sidecars.clone(),
        None,
    );
    Ok(ReadView {
        engine: Box::new(DynamicEngine::<T> {
            index,
            receipts,
            auto_pin: Default::default(),
            pin_report: Default::default(),
        }),
        object_store: saved.store,
        payloads: files,
        graph_bytes: graph.bytes,
        metadata_bytes: saved.metadata.len() as u64,
    })
}
