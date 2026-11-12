/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Load/build a working graph and publish its immutable provenance object.
//! The working key stays attached to sidecar receipts, while payload identities
//! remain content addressed and independent of the mutable cache lookup record.
use super::super::{
    snapshot::Payloads,
    view::{DynamicEngine, ReadView},
};
use crate::{
    config::ResolvedRunConfig,
    data::{BaseElement, LoadedBase},
    graph::cache::CachePlan,
    Result,
};
use std::collections::BTreeMap;

pub(super) fn prepare<T: BaseElement + 'static>(
    config: &ResolvedRunConfig,
    mut data: LoadedBase<T>,
    plan: CachePlan,
    base_sha256: &str,
    load: impl FnOnce(
        &ResolvedRunConfig,
        &mut LoadedBase<T>,
        CachePlan,
    ) -> Result<crate::graph::LoadedGraph<T>>,
) -> Result<ReadView>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    let points = data.num_points;
    let mut graph = load(config, &mut data, plan)?;
    crate::search::bind_rows(&graph.index, config)?;
    let record = graph.generation;
    let (bytes, digest) = crate::storage::put(&graph.store, "cache.json", |w| {
        std::io::Write::write_all(w, &record.encoded)?;
        Ok(())
    })?;
    let files = Payloads {
        supplier: None,
        graph_sha256: record.manifest.graph_sha256,
        metadata_sha256: record.manifest.metadata_sha256,
        provenance_sha256: digest,
        provenance_bytes: bytes,
        base_sha256: base_sha256.to_owned(),
        sidecars: BTreeMap::new(),
        external_ids: None,
        labels: None,
    };
    let receipts = crate::sidecars::track(
        &mut graph.index,
        graph.store.clone(),
        Some(graph.working_key),
        BTreeMap::new(),
        crate::sidecars::derived_key(config, points)?,
    );
    Ok(ReadView {
        engine: Box::new(DynamicEngine::<T> {
            auto_pin: Default::default(),
            pin_report: Default::default(),
            index: graph.index,
            receipts,
        }),
        object_store: graph.store.canonicalize()?,
        payloads: files,
        graph_bytes: graph.graph_bytes,
        metadata_bytes: graph.metadata_bytes,
    })
}
